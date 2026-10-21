"""The JAX partition of the Tome's network: build the connectome and integrate it one window at a time.

``build(seed)`` compiles ``params`` into padded tables. ``make_window(net)`` returns a jitted
``window(state, ext[W, N], glucose) -> (state, spikes[W, N])`` that advances W ms of the 1-ms master clock.

Integration (design §6): every compartment uses exponential Euler with Jacobi coupling (conductances and neighbour
voltages frozen over a 0.25-ms substep), so it cannot blow up at any dt. Synaptic conductances decay exactly.
Delivery is event-driven: each ms, the cells that spiked are compacted and their padded out-edges scattered into
a 32-ms future-conductance ring. Every random draw is keyed by absolute tick (``jax.random.fold_in``), so the
result does not depend on how time is windowed.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import params as PR

RING = 32          # future-conductance ring (ms): longest axonal delay must stay below it
PROBE_N = 7        # per-ms probe: g_exc, g_inh (L2/3 PCs), DA, NE, 5-HT, ACh, prediction-error F (S/T L2/3)
HIST = 16          # spike-history ring (ms) for the Graupner-Brunel presynaptic calcium delay
GB_DELAY = 14      # D = 13.7 ms, rounded to the clock
# Graupner & Brunel 2012 "DP" cortical set; theta_p = 1.3 (the spec's 1.6 breaks frequency dependence)
GB = dict(Cpre=1.0, Cpost=2.0, tauCa=20.0, thd=1.0, thp=1.3, gd=200.0, gp=321.808, tau=150e3, rs=0.5)
VOGELS = dict(eta=0.01, rho0=0.002, tau=20.0, wmax=5.0)     # on the efficacy multiplier; rho0 = 2 Hz for L2/3 PCs
TM = dict(U=(0.5, 0.1), tau_d=(500.0, 100.0), tau_f=(20.0, 1000.0))  # depressing, facilitating
NEUROMOD = ("DA", "NE", "5HT", "ACh")
NEUROMOD_SRC = ("VTA", "LC", "DRN", "NBM")
NEUROMOD_TAU = (200.0, 500.0, 1000.0, 300.0)
# De Pitta 2009 with v_ER = 2.0 uM/s (the AM set's 0.9 saturates; 2.0 puts the cortical drive range in the
# frequency-coding regime: Ca2+ event frequency rises with neuronal activity, gate G3a).
ASTRO = dict(rC=6.0, rL=0.11, CT=2.0, c1=0.185, vER=2.0, KER=0.1, d1=0.13, d2=1.049, d3=0.9434, d5=0.08234,
             a2=0.2, v_beta=0.5, v_delta=0.02, kappa_delta=1.5, K_delta=0.1, v3K=2.0, KD=0.7, K3=1.0, r5P=0.04,
             glu_tau=50.0, glu_gain=0.1, gamma_tau=1000.0, gamma_jump=0.6, gamma_theta=0.5)
# K_ATP g_max: 0.6 nS on cortical excitatory cells puts the cortex at about half its baseline at Lumen 0 (G3b: weak,
# not comatose); 4 nS on the dopamine cells, whose firing fails first when energy runs out (Liss et al. 2005).
ENERGY = dict(k_sup=1.0, k_rest=0.06, k_syn=0.01, k_spike=0.004, katp_gmax=0.6, katp_gmax_da=4.0, katp_half=0.55,
              nad_tau=60e3)


@dataclass
class Net:
    """The compiled network: static tables (numpy) plus index metadata."""

    pops: list
    start: dict
    size: dict
    n: int
    n3: int                      # 3-compartment cells occupy [0, n3)
    n2: int                      # 2-compartment cells occupy [n3, n3 + n2)
    n_slot: int
    tables: dict
    point: dict                  # per-cell parameter vectors (all cells)
    n_gb: int
    n_vog: int
    n_astro: int
    s_max: int
    digest: str
    seed: int
    mu: np.ndarray = field(repr=False, default=None)
    comp_base: np.ndarray = field(repr=False, default=None)

    def slot(self, cell: int, comp: int, rec: int) -> int:
        """Receptor slot of a cell's compartment (point cells: soma; 2-comp cells: soma/basal -> soma)."""
        nc = 3 if cell < self.n3 else 2 if cell < self.n3 + self.n2 else 1
        if nc == 1:
            comp = 0
        elif nc == 2:
            comp = 0 if comp in (PR.SOMA, PR.BASAL) else 1
        return int(self.comp_base[cell] + comp * PR.N_REC + rec)

    def idx(self, name: str) -> slice:
        """Index range of a population, or of a named assembly in params.ASSEMBLIES."""
        if name in PR.ASSEMBLIES:
            pop, a, b = PR.ASSEMBLIES[name]
            s = self.start[pop]
            return slice(s + a, s + b)
        if "[" in name:                                   # "pop[a:b]": a sub-range of a population
            pop, rng = name[:-1].split("[")
            a, b = (int(x) for x in rng.split(":"))
            return slice(self.start[pop] + a, self.start[pop] + b)
        return slice(self.start[name], self.start[name] + self.size[name])


def _range(start: dict, size: dict, spec: str) -> tuple[int, int]:
    if "[" in spec:
        pop, rng = spec[:-1].split("[")
        a, b = (int(x) for x in rng.split(":"))
        return start[pop] + a, b - a
    return start[spec], size[spec]


CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "connectome_calibrated.toml"


def inhibitory_onto_l23() -> list[int]:
    """Indices of the inhibitory projections onto L2/3 PCs (the gains G5 balances)."""
    return [k for k, pr in enumerate(PR.projections())
            if pr.dst.endswith(".L23") and pr.recs[0] in (PR.GABA_A, PR.GABA_AS)]


def load_calibration(path: Path = CALIBRATION_FILE) -> dict:
    """Tonic currents and the L2/3 inhibitory gain fitted by G5 (empty if not calibrated yet)."""
    if not path.exists():
        return {"mu": {}, "inh_gain": 1.0, "sd_scale": 1.0}
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return {"mu": dict(raw.get("mu", {})), "inh_gain": float(raw.get("inhibitory_gain_onto_l23", 1.0)),
            "sd_scale": float(raw.get("background_sd_scale", 1.0))}


def build(seed: int = 0, s_max: int = 96, mu: dict | None = None, gains: dict | None = None,
          pops: list | None = None, projs: list | None = None, calibrated: bool = True) -> Net:
    """Compile the connectome. ``mu`` overrides population tonic currents; ``gains`` scales projections by
    their index in ``params.projections()`` (both are what G5 calibration fits). ``pops``/``projs`` build a
    small custom network instead (cell-level gates)."""
    sd_scale = 1.0
    if calibrated and pops is None:
        # the calibration fills in whatever the caller did not set explicitly
        cal = load_calibration()
        if mu is None:
            mu = cal["mu"] or None
        merged = {k: cal["inh_gain"] for k in inhibitory_onto_l23()} if cal["inh_gain"] != 1.0 else {}
        merged.update(gains or {})
        gains = merged or None
        sd_scale = cal["sd_scale"]
    pops = PR.populations() if pops is None else pops
    proj_list = PR.projections() if projs is None else projs
    start, size, i = {}, {}, 0
    for p in pops:
        start[p.name], size[p.name] = i, p.n
        i += p.n
    n = i
    n3 = sum(p.n for p in pops if p.kind == "L23")
    n2 = sum(p.n for p in pops if p.kind == "L5")
    rng = np.random.default_rng(seed)

    def ncomp(cell: int) -> int:
        return 3 if cell < n3 else 2 if cell < n3 + n2 else 1

    comp_base = np.zeros(n + 1, np.int64)
    for c in range(n):
        comp_base[c + 1] = comp_base[c] + ncomp(c) * PR.N_REC
    n_slot = int(comp_base[n])

    def slot(cell: int, comp: int, rec: int) -> int:
        nc = ncomp(cell)
        if nc == 1:
            comp = 0
        elif nc == 2:
            comp = 0 if comp in (PR.SOMA, PR.BASAL) else 1
        return int(comp_base[cell] + comp * PR.N_REC + rec)

    edges = []          # (pre, slot, w, delay, stp, gb_id, vog_id, mod)
    gb_pre, gb_post, vog_pre, vog_post = [], [], [], []
    for k, pr in enumerate(proj_list):
        s0, ns = _range(start, size, pr.src)
        t0, nt = _range(start, size, pr.dst)
        g = (gains or {}).get(k, 1.0)
        for j in range(nt):
            post = t0 + j
            pres = rng.choice(ns, size=min(pr.K, ns), replace=False) + s0
            for pre in pres:
                if pre == post:
                    continue
                d = int(rng.integers(pr.delay[0], pr.delay[1] + 1))
                gid = vid = -1
                if pr.plastic == "gb":
                    gid = len(gb_pre)
                    gb_pre.append(pre)
                    gb_post.append(post)
                elif pr.plastic == "vogels":
                    vid = len(vog_pre)
                    vog_pre.append(pre)
                    vog_post.append(post)
                jitter = rng.uniform(0.7, 1.3)
                comp_is_tuft = pr.comp == PR.APICAL and ncomp(post) > 1
                for r in pr.recs:
                    scale = 1.0
                    if r == PR.NMDA:
                        ratios = PR.L23["nmda_ratio"] if ncomp(post) == 3 else PR.L5["nmda_ratio"]
                        scale = ratios[-1] if comp_is_tuft else ratios[0]
                    elif r == PR.GABA_B:
                        scale = 0.3
                    edges.append((pre, slot(post, pr.comp, r), pr.w * g * scale * jitter, d, pr.stp, gid, vid,
                                  1 if pr.mod == "ach" else 0))
    by_pre: list[list] = [[] for _ in range(n)]
    for e in edges:
        by_pre[e[0]].append(e)
    kmax = max(1, max(len(b) for b in by_pre))
    T = dict(slot=np.full((n, kmax), n_slot, np.int32), w=np.zeros((n, kmax), np.float32),
             d=np.ones((n, kmax), np.int32), stp=np.zeros((n, kmax), np.int32),
             gb=np.full((n, kmax), -1, np.int32), vog=np.full((n, kmax), -1, np.int32),
             mod=np.zeros((n, kmax), np.int32))
    for i, b in enumerate(by_pre):
        for k, (_, sl, w, d, stp, gid, vid, mod) in enumerate(b):
            T["slot"][i, k], T["w"][i, k], T["d"][i, k], T["stp"][i, k] = sl, w, d, stp
            T["gb"][i, k], T["vog"][i, k], T["mod"][i, k] = gid, vid, mod
    if int(T["d"].max()) >= RING:
        raise ValueError("an axonal delay does not fit the future-conductance ring")
    # (a network without plastic synapses keeps one inert placeholder so the tables are never empty)
    gb_pre, gb_post = (gb_pre, gb_post) if gb_pre else ([0], [0])
    vog_pre, vog_post = (vog_pre, vog_post) if vog_pre else ([0], [0])
    T.update(gb_pre=np.array(gb_pre, np.int32), gb_post=np.array(gb_post, np.int32),
             vog_pre=np.array(vog_pre, np.int32), vog_post=np.array(vog_post, np.int32))

    # per-cell parameters
    pt = {f: np.zeros(n, np.float32) for f in PR.POINT_FIELDS}
    flags = {k: np.zeros(n, bool) for k in ("inh", "cortical", "orx", "serotonin_target", "tc", "katp")}
    katp_g = np.zeros(n, np.float32)
    region = np.zeros(n, np.int32)
    regions = ("S", "T", "P", "I", "thal", "hippo", "nuclei")
    mu_v = np.zeros(n, np.float32)
    sd_v = np.zeros(n, np.float32)
    for p in pops:
        sl = slice(start[p.name], start[p.name] + p.n)
        if p.kind == "L23":
            src = PR.L23
            vals = dict(C=src["C"][0], gL=src["gL"][0], EL=src["EL"], VT=src["VT"], DT=src["DT"], a=src["a"],
                        b=src["b"], tau_w=src["tau_w"], Vr=src["Vr"], t_ref=src["t_ref"], I_bias=0.0, g_T=0.0)
        elif p.kind == "L5":
            src = PR.L5
            vals = dict(C=src["C"][0], gL=src["gL"][0], EL=src["EL"], VT=src["VT"], DT=src["DT"], a=src["a"],
                        b=src["b"], tau_w=src["tau_w"], Vr=src["Vr"], t_ref=src["t_ref"], I_bias=0.0, g_T=0.0)
        else:
            vals = dict(zip(PR.POINT_FIELDS, PR.POINT_KINDS[p.kind]))
        for f in PR.POINT_FIELDS:
            pt[f][sl] = vals[f]
        flags["inh"][sl] = p.sign == "I"
        flags["cortical"][sl] = p.cortical
        # K_ATP: cortical excitatory cells and the dopamine cells (Liss et al. 2005) slow down as ATP fails
        flags["katp"][sl] = (p.cortical and p.sign == "E") or p.name == "VTA"
        katp_g[sl] = ENERGY["katp_gmax_da"] if p.name == "VTA" else ENERGY["katp_gmax"]
        flags["orx"][sl] = p.name == "ORX"
        flags["tc"][sl] = vals["g_T"] > 0
        region[sl] = regions.index(p.region)
        mu_v[sl] = (mu or {}).get(p.name, p.mu)
        sd_v[sl] = PR.OU_SD[p.sign] * sd_scale
    for name in ("P.puppet.elara", "P.puppet.marcus", "P.bend"):
        pop, a, b = PR.ASSEMBLIES[name]
        if pop in start:
            flags["serotonin_target"][start[pop] + a:start[pop] + b] = True

    # astrocyte domains: one per 12 L2/3 PCs per column, plus 8 over CA3 + CA1
    astro_of = np.full(n, -1, np.int32)
    dom = 0
    for c in PR.COLUMNS:
        if f"{c}.L23" not in start:
            continue
        s0, nn = start[f"{c}.L23"], size[f"{c}.L23"]
        for k in range(nn):
            astro_of[s0 + k] = dom + k // 12
        dom += max(1, nn // 12)
    hip = [c for name in ("CA3", "CA1") if name in start for c in range(start[name], start[name] + size[name])]
    if hip:
        per = math.ceil(len(hip) / 8)
        for k, cell in enumerate(hip):
            astro_of[cell] = dom + k // per
        dom += 8
    dom = max(dom, 1)
    nm_src = np.full(n, -1, np.int32)
    for m, src in enumerate(NEUROMOD_SRC):
        if src in start:
            nm_src[start[src]:start[src] + size[src]] = m
    point = dict(pt, **flags, region=region, astro_of=astro_of, nm_src=nm_src, sd=sd_v, katp_g=katp_g)

    h = hashlib.sha256()
    for k in sorted(T):
        h.update(k.encode())
        h.update(np.ascontiguousarray(T[k]).tobytes())
    h.update(mu_v.tobytes())
    return Net(pops=pops, start=start, size=size, n=n, n3=n3, n2=n2, n_slot=n_slot, tables=T, point=point,
               n_gb=len(gb_pre), n_vog=len(vog_pre), n_astro=dom, s_max=s_max, digest=h.hexdigest(), seed=seed,
               mu=mu_v, comp_base=comp_base)


# ====================================================================== state
def init_state(net: Net, key):
    import jax
    import jax.numpy as jnp

    f = jnp.float32
    n, n3, n2 = net.n, net.n3, net.n2
    k1, k2, k3 = jax.random.split(key, 3)
    EL = jnp.asarray(net.point["EL"])
    return dict(
        V=EL + jax.random.uniform(k1, (n,), f, 0.0, 6.0),
        Vb=jnp.full((n3,), PR.L23["EL"], f), Va=jnp.full((n3,), PR.L23["EL"], f),
        Vt=jnp.full((n2,), PR.L5["EL"], f),
        w=jnp.zeros(n, f), refr=jnp.zeros(n, f), hT=jnp.full(n, 0.5, f), ou=jnp.zeros(n, f),
        # dCaAP gates (L2/3 tuft) and BAC (L5 / CA1 tuft)
        dm=jnp.zeros(n3, f), dh=jnp.ones(n3, f), dj=jnp.ones(n3, f), du=jnp.zeros(n3, f), dref=jnp.zeros(n3, f),
        bap3=jnp.zeros(n3, f), bap2=jnp.zeros(n2, f),
        bac1=jnp.zeros(n2, f), bac2=jnp.zeros(n2, f), bacref=jnp.zeros(n2, f),
        G=jnp.zeros(net.n_slot + 1, f), Q=jnp.zeros((RING, net.n_slot + 1), f),
        hist=jnp.zeros((HIST, n), f),
        x=jnp.ones((2, n), f), u=jnp.zeros((2, n), f),
        cpre=jnp.zeros(n, f), cpost=jnp.zeros(n, f),
        rho=jnp.clip(0.5 + 0.1 * jax.random.normal(k2, (net.n_gb,), f), 0.05, 0.95),
        wv=jnp.ones(net.n_vog, f), xpre=jnp.zeros(n, f), xpost=jnp.zeros(n, f),
        astro=jnp.stack([jnp.full(net.n_astro, 0.1, f), jnp.full(net.n_astro, 0.05, f),
                         jnp.full(net.n_astro, 0.8, f), jnp.zeros(net.n_astro, f), jnp.zeros(net.n_astro, f)]),
        atp=jnp.full(n, 0.99, f), nad=jnp.asarray(0.99, f),
        nm=jnp.zeros(4, f), da_tonic=jnp.asarray(0.0, f),
        overflow=jnp.asarray(0, jnp.int32), t=jnp.asarray(0, jnp.int32), key=k3,
        mu=jnp.asarray(net.mu, f), sd=jnp.asarray(net.point["sd"], f),
    )


# ====================================================================== dynamics
def gb_drho(rho, c, gp_scale=1.0):
    """Graupner-Brunel d(rho)/dt per ms for calcium ``c`` (bistable, calcium-threshold LTP/LTD)."""
    return (-rho * (1 - rho) * (GB["rs"] - rho) + GB["gp"] * gp_scale * (1 - rho) * (c >= GB["thp"])
            - GB["gd"] * rho * (c >= GB["thd"])) / GB["tau"]


def tm_step(x, u, s):
    """One ms of Tsodyks-Markram STP for both synapse classes ([2, N] state, s = spikes this ms).

    Between spikes u decays to 0 (tau_F) and x recovers to 1 (tau_D). At a spike: u+ = u + U(1 - u), the release
    fraction is u+ * x-, then x -= release. Returns (x, u, release) with release normalised by U, so a rested
    synapse delivers exactly its weight."""
    import jax.numpy as jnp
    U = jnp.asarray(TM["U"], jnp.float32)[:, None]
    x = 1.0 - (1.0 - x) * jnp.exp(-1.0 / jnp.asarray(TM["tau_d"], jnp.float32))[:, None]
    u = u * jnp.exp(-1.0 / jnp.asarray(TM["tau_f"], jnp.float32))[:, None]
    u_plus = u + U * (1.0 - u)
    rel = u_plus * x
    spiking = s[None, :] > 0
    return jnp.where(spiking, x - rel, x), jnp.where(spiking, u_plus, u), rel / U


def astro_step(astro, dom_spk, dt_ms: float = 1.0):
    """``dt_ms`` of the De Pitta 2009 astrocyte (Li-Rinzel Ca2+ with IP3 metabolism) for every domain.

    ``astro`` rows: IP3, Ca, h, gliotransmitter Gamma, glutamate spill-over trace. ``dom_spk``: spikes of the
    domain's neurons per ms. Gamma jumps at each upward crossing of Ca = 0.5 uM and decays with tau 1 s. The quiet
    interval calls it with 20-ms steps."""
    import jax.numpy as jnp
    A = ASTRO
    ip3, C, h, gm, glu = astro
    k = math.exp(-dt_ms / A["glu_tau"])
    glu = glu * k + A["glu_gain"] * dom_spk * (1.0 - k) * A["glu_tau"] if dt_ms > 1.0 else \
        glu * k + A["glu_gain"] * dom_spk
    m_inf = ip3 / (ip3 + A["d1"])
    n_inf = C / (C + A["d5"])
    free = A["CT"] - (1 + A["c1"]) * C
    dC = (A["rC"] * m_inf ** 3 * n_inf ** 3 * h ** 3 * free + A["rL"] * free
          - A["vER"] * C ** 2 / (C ** 2 + A["KER"] ** 2))
    Q2 = A["d2"] * (ip3 + A["d1"]) / (ip3 + A["d3"])
    dh = A["a2"] * (Q2 * (1 - h) - C * h)
    dI = (A["v_beta"] * jnp.tanh(glu) + A["v_delta"] / (1 + ip3 / A["kappa_delta"]) * C ** 2 /
          (C ** 2 + A["K_delta"] ** 2) - A["v3K"] * C ** 4 / (C ** 4 + A["KD"] ** 4) * ip3 / (ip3 + A["K3"])
          - A["r5P"] * ip3)
    s = 1e-3 * dt_ms
    C_n = jnp.clip(C + s * dC, 0.0, A["CT"])
    up = (C < A["gamma_theta"]) & (C_n >= A["gamma_theta"])
    gm = gm * math.exp(-dt_ms / A["gamma_tau"])
    gm = jnp.where(up, gm + A["gamma_jump"] * (1 - gm), gm)
    return jnp.stack([jnp.maximum(ip3 + s * dI, 0.0), C_n, jnp.clip(h + s * dh, 0.0, 1.0), gm, glu])


def mg_block(v):
    """Jahr-Stevens NMDA Mg2+ block, [Mg] = 1 mM."""
    import jax.numpy as jnp
    return 1.0 / (1.0 + jnp.exp(-0.062 * v) / 3.57)


def make_window(net: Net, dt: float = 0.25, flags: dict | None = None, dcaap: dict | None = None,
                record: bool = False, max_w: int | None = None):
    """Return a jitted ``window(state, ext[W, N], glucose, gclamp=None, gbus=None)``
    -> ``(state, (spikes[W, N], probe[W, 2]))``. ``gbus[W, n_slot + 1]`` is conductance arriving over the bus.

    ``probe[W, 7]`` holds, each ms, the mean excitatory and inhibitory conductance (nS) on the L2/3 PCs, the DA,
    NE, 5-HT and ACh concentrations, and the precision-weighted prediction error F of the S and T L2/3 cells."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    FL = dict(plasticity=True, glia=True, energy=True, inhibition=True, noise=True)
    FL.update(flags or {})
    DC = dict(PR.DCAAP)
    DC.update(dcaap or {})
    f = jnp.float32
    n, n3, n2, n_slot = net.n, net.n3, net.n2, net.n_slot
    nsub = int(round(1.0 / dt))
    T = {k: jnp.asarray(v) for k, v in net.tables.items()}
    p = {k: jnp.asarray(v) for k, v in net.point.items()}
    if not FL["inhibition"]:   # the G4 mutant: every inhibitory synapse silenced
        T["w"] = jnp.where(p["inh"][:, None], 0.0, T["w"])

    # receptor decay per substep, laid out like the slots: [cell comps x receptors] flattened, plus the sink
    ncomp = np.where(np.arange(n) < n3, 3, np.where(np.arange(n) < n3 + n2, 2, 1))
    decay = np.concatenate([np.tile(np.exp(-dt / np.array(PR.TAU_REC)), int(ncomp.sum())), [0.0]])
    decay = jnp.asarray(decay, f)
    E = jnp.asarray(PR.E_REC, f)
    o3, o2 = 0, n3 * 3 * PR.N_REC
    o1 = o2 + n2 * 2 * PR.N_REC
    L3, L5 = PR.L23, PR.L5
    k_ca = math.exp(-1.0 / GB["tauCa"])
    k_v = math.exp(-1.0 / VOGELS["tau"])
    alpha_v = 2.0 * VOGELS["rho0"] * VOGELS["tau"]
    nm_k = jnp.exp(-1.0 / jnp.asarray(NEUROMOD_TAU, f))
    ou_k = math.exp(-1.0 / PR.OU_TAU)
    ou_s = math.sqrt(1.0 - ou_k ** 2)
    astro_of = p["astro_of"]
    in_dom = astro_of >= 0
    dom_idx = jnp.maximum(astro_of, 0)
    dom_n = jax.ops.segment_sum(in_dom.astype(f), dom_idx, net.n_astro)
    nm_src = p["nm_src"]
    is_P = p["region"] == 2
    n_st = int(sum(pp.n for pp in net.pops if pp.kind == "L23" and pp.region in ("S", "T")))   # S and T L2/3 lead

    def syn(Gc, v, nmda_gain):
        """Total synaptic conductance and conductance x reversal for one compartment ([cells, 5] slots)."""
        gn = Gc[:, 1] * mg_block(v) * nmda_gain
        g = Gc[:, 0] + gn + Gc[:, 2] + Gc[:, 3] + Gc[:, 4]
        ge = Gc[:, 0] * E[0] + gn * E[1] + (Gc[:, 2] + Gc[:, 3]) * E[2] + Gc[:, 4] * E[4]
        return g, ge

    def relax(v, g, num, c):
        vinf = num / g
        return vinf + (v - vinf) * jnp.exp(-dt * g / c)

    def substep(st, G, I_soma, katp, VT, nmda_gain, gclamp):
        V = st["V"]
        G3 = G[o3:o2].reshape(n3, 3, PR.N_REC)
        G2 = G[o2:o1].reshape(n2, 2, PR.N_REC)
        G1 = G[o1:n_slot].reshape(n - n3 - n2, PR.N_REC)
        g_k = katp
        expo = p["gL"] * p["DT"] * jnp.exp(jnp.minimum((V - VT) / p["DT"], PR.EXP_CLAMP))
        # ---- L2/3: soma, basal, apical tuft with the dCaAP
        Vs, Vb, Va = V[:n3], st["Vb"], st["Va"]
        ng3 = nmda_gain[:n3]
        gs, ges = syn(G3[:, 0], Vs, ng3)
        gb, geb = syn(G3[:, 1], Vb, ng3)
        ga, gea = syn(G3[:, 2], Va, ng3)
        ga = ga + gclamp   # excitatory (E = 0 mV) conductance clamp on the tuft
        u_in = G3[:, 2, 0] + G3[:, 2, 1] * mg_block(Va) * ng3 + gclamp
        du = st["du"] + (1 - math.exp(-dt / DC["tau_u"])) * (u_in - st["du"])
        active = (st["dref"] <= 0) | (st["dref"] > DC["refractory"] / 2)
        m_inf = jax.nn.sigmoid((Va - DC["theta"]) / 2.5) * active
        dm = st["dm"] + (1 - math.exp(-dt / DC["tau_m"])) * (m_inf - st["dm"])
        h_inf = jnp.exp(-jnp.maximum(0.0, du / DC["u_th"] - 1.0) / DC["tau_K"])
        tau_h = jnp.where(h_inf < st["dh"], DC["tau_h_fall"], DC["tau_h_rec"])
        dh = st["dh"] + (1 - jnp.exp(-dt / tau_h)) * (h_inf - st["dh"])
        dj = st["dj"] + (1 - math.exp(-dt / DC["tau_j"])) * (jax.nn.sigmoid(-(Va + 5.0) / 4.0) - st["dj"])
        g_d = DC["g_d"] * dm * dh * dj
        onset = (st["dm"] < 0.5) & (dm >= 0.5)
        dref = jnp.where(onset, DC["refractory"], jnp.maximum(st["dref"] - dt, 0.0))
        g_kv3 = L3["g_kv"] / (1 + jnp.exp(-(Va + 35.0) / 5.0))
        k3 = g_k[:n3]
        Vs_n = relax(Vs, L3["gL"][0] + gs + L3["g_sb"] + L3["g_sa"] + k3,
                     L3["gL"][0] * L3["EL"] + ges + L3["g_sb"] * Vb + L3["g_sa"] * Va - 90 * k3 + expo[:n3]
                     - st["w"][:n3] + I_soma[:n3], L3["C"][0])
        Vb_n = relax(Vb, L3["gL"][1] + gb + L3["g_sb"] + k3,
                     L3["gL"][1] * L3["EL"] + geb + L3["g_sb"] * Vs - 90 * k3, L3["C"][1])
        Va_n = relax(Va, L3["gL"][2] + ga + L3["g_sa"] + g_d + g_kv3 + k3,
                     L3["gL"][2] * L3["EL"] + gea + L3["g_sa"] * Vs + g_d * DC["E_Ca"] - 90 * (g_kv3 + k3)
                     + st["bap3"], L3["C"][2])
        # ---- L5 ET / CA1: soma and tuft with BAC firing
        s2 = slice(n3, n3 + n2)
        V5, Vt = V[s2], st["Vt"]
        ng2 = nmda_gain[s2]
        g5, ge5 = syn(G2[:, 0], V5, ng2)
        gt, get_ = syn(G2[:, 1], Vt, ng2)
        g_kv2 = L5["g_kv"] / (1 + jnp.exp(-(Vt + 35.0) / 5.0))
        I_bac = L5["bac_amp"] * 1.354 * (st["bac1"] - st["bac2"])
        k2 = g_k[s2]
        V5_n = relax(V5, L5["gL"][0] + g5 + L5["g_sa"] + k2,
                     L5["gL"][0] * L5["EL"] + ge5 + L5["g_sa"] * Vt - 90 * k2 + expo[s2] - st["w"][s2]
                     + I_soma[s2], L5["C"][0])
        Vt_n = relax(Vt, L5["gL"][1] + gt + L5["g_sa"] + g_kv2 + k2,
                     L5["gL"][1] * L5["EL"] + get_ + L5["g_sa"] * V5 - 90 * (g_kv2 + k2) + I_bac + st["bap2"],
                     L5["C"][1])
        bac_on = (Vt < L5["bac_theta"]) & (Vt_n >= L5["bac_theta"]) & (st["bacref"] <= 0)
        bac1 = jnp.where(bac_on, 1.0, st["bac1"] * math.exp(-dt / 25.0))
        bac2 = jnp.where(bac_on, 1.0, st["bac2"] * math.exp(-dt / 2.0))
        bacref = jnp.where(bac_on, L5["bac_refractory"], jnp.maximum(st["bacref"] - dt, 0.0))
        # ---- point cells (with the T-type current where g_T > 0)
        s1 = slice(n3 + n2, n)
        Vp = V[s1]
        g1, ge1 = syn(G1, Vp, nmda_gain[s1])
        mT = 1.0 / (1.0 + jnp.exp(-(Vp + 57.0) / 6.2))
        hT = st["hT"][s1]
        gT = p["g_T"][s1] * mT * mT * hT
        k1 = g_k[s1]
        Vp_n = relax(Vp, p["gL"][s1] + g1 + gT + k1,
                     p["gL"][s1] * p["EL"][s1] + ge1 + gT * 120.0 - 90 * k1 + expo[s1] - st["w"][s1] + I_soma[s1],
                     p["C"][s1])
        hT_inf = 1.0 / (1.0 + jnp.exp((Vp + 81.0) / 4.0))
        tau_hT = 8.0 + 60.0 / (1.0 + jnp.exp((Vp + 78.0) / 3.0))
        hT_n = hT_inf + (hT - hT_inf) * jnp.exp(-dt / tau_hT)
        # ---- spikes, reset, adaptation, refractoriness
        Vn = jnp.concatenate([Vs_n, V5_n, Vp_n])
        inref = st["refr"] > 0
        Vn = jnp.where(inref, p["Vr"], Vn)
        spk = (Vn >= PR.SPIKE_V) & ~inref
        Vn = jnp.where(spk, p["Vr"], Vn)
        winf = p["a"] * (Vn - p["EL"])
        w = winf + (st["w"] - winf) * jnp.exp(-dt / jnp.maximum(p["tau_w"], 1.0)) + p["b"] * spk
        refr = jnp.where(spk, p["t_ref"], jnp.maximum(st["refr"] - dt, 0.0))
        bap3 = jnp.where(spk[:n3], L3["bap"], st["bap3"] * math.exp(-dt))
        bap2 = jnp.where(spk[s2], L5["bap"], st["bap2"] * math.exp(-dt))
        st = dict(st, V=Vn, Vb=Vb_n, Va=Va_n, Vt=Vt_n, w=w, refr=refr, dm=dm, dh=dh, dj=dj, du=du, dref=dref,
                  bap3=bap3, bap2=bap2, bac1=bac1, bac2=bac2, bacref=bacref,
                  hT=st["hT"].at[s1].set(hT_n))
        return st, spk

    def ms_step(st, inp):
        ext, glucose, gclamp, gbus = inp
        t = st["t"]
        q = t % RING
        G = st["G"] + st["Q"][q] + gbus      # local deliveries plus conductance arriving over the spike bus
        Qr = st["Q"].at[q].set(0.0)
        # background: OU noise keyed by absolute tick, plus the population's tonic mean
        if FL["noise"]:
            xi = jax.random.normal(jax.random.fold_in(st["key"], t), (n,), f)
            ou = st["ou"] * ou_k + st["sd"] * ou_s * xi
        else:
            ou = st["ou"]
        nm = st["nm"]
        I = st["mu"] + ou + p["I_bias"] + ext + p["orx"] * 260.0 * (1.0 - glucose) - p["serotonin_target"] * 20.0 * nm[2]
        VT = p["VT"] - 2.0 * nm[1] * p["cortical"]
        atp = st["atp"]
        katp = p["katp"] * p["katp_g"] / (1.0 + (atp / ENERGY["katp_half"]) ** 4) if FL["energy"] \
            else jnp.zeros(n, f)
        gamma = st["astro"][3]
        nmda_gain = (1.0 + 0.5 * jnp.where(in_dom, gamma[dom_idx], 0.0)) * (1.0 + 0.3 * st["da_tonic"] * is_P)
        spk_ms = jnp.zeros(n, bool)
        va_max = jnp.full(n3, -200.0, f)
        for _ in range(nsub):
            G = G * decay
            st, spk = substep(st, G, I, katp, VT, nmda_gain, gclamp)
            spk_ms = spk_ms | spk
            va_max = jnp.maximum(va_max, st["Va"])
        s = spk_ms.astype(f)
        # Tsodyks-Markram STP per presynaptic cell: u decays to 0, x recovers to 1; at a spike u+ = u + U(1-u),
        # release = u+ * x-, then x -= release.
        x, u, rel = tm_step(st["x"], st["u"], s)
        # event-driven delivery: compact the spiking cells in chunks of s_max and scatter their out-edges into the
        # future ring. A synchronous volley simply takes more chunks, so no spike is ever dropped.
        spk_i = spk_ms.astype(jnp.int32)
        count = jnp.sum(spk_i)
        rank = jnp.cumsum(spk_i) - 1
        rho_s, wv_s = st["rho"], st["wv"]

        def deliver(k, acc):
            Qa, done = acc
            sel = spk_ms & (rank >= k * net.s_max) & (rank < (k + 1) * net.s_max)
            idx = jnp.nonzero(sel, size=net.s_max, fill_value=n)[0]
            valid = (idx < n)[:, None]
            ii = jnp.minimum(idx, n - 1)
            gid, vid = T["gb"][ii], T["vog"][ii]
            eff = jnp.where(gid >= 0, 0.4 + 1.2 * rho_s[jnp.maximum(gid, 0)], 1.0)
            eff = eff * jnp.where(vid >= 0, wv_s[jnp.maximum(vid, 0)], 1.0)
            stp = T["stp"][ii]
            fac = jnp.where(stp == 1, rel[0, ii][:, None], jnp.where(stp == 2, rel[1, ii][:, None], 1.0))
            ach = jnp.where(T["mod"][ii] == 1, 1.0 - 0.5 * nm[3], 1.0)
            inc = T["w"][ii] * eff * fac * ach * valid
            Qa = Qa.at[(t + T["d"][ii]) % RING, T["slot"][ii]].add(inc)
            return Qa, done + jnp.sum(valid.astype(jnp.int32))

        n_chunks = (count + net.s_max - 1) // net.s_max
        Qr, delivered = lax.fori_loop(0, n_chunks, deliver, (Qr, jnp.asarray(0, jnp.int32)))
        overflow = st["overflow"] + (count - delivered)
        st = dict(st, x=x, u=u, ou=ou, G=G, Q=Qr, overflow=overflow)
        # Graupner-Brunel on plastic excitatory synapses (presynaptic calcium arrives D ms late)
        hist = st["hist"].at[t % HIST].set(s)
        if FL["plasticity"]:
            pre_d = hist[(t - GB_DELAY) % HIST]
            cpre = st["cpre"] * k_ca + GB["Cpre"] * pre_d * (1.0 + 0.5 * jnp.where(in_dom, gamma[dom_idx], 0.0))
            cpost = st["cpost"] * k_ca + GB["Cpost"] * s
            c = cpre[T["gb_pre"]] + cpost[T["gb_post"]]
            rho = jnp.clip(st["rho"] + gb_drho(st["rho"], c, st["nad"]), 0.0, 1.0)
            # Vogels iSTDP on PV/SST -> L2/3 PC synapses: pre spike w += eta (x_post - alpha); post spike += eta x_pre
            xpre = st["xpre"] * k_v + s
            xpost = st["xpost"] * k_v + s
            vp, vq = T["vog_pre"], T["vog_post"]
            dw = VOGELS["eta"] * (s[vp] * (xpost[vq] - alpha_v) + s[vq] * xpre[vp])
            wv = jnp.clip(st["wv"] + dw, 0.0, VOGELS["wmax"])
            st = dict(st, cpre=cpre, cpost=cpost, rho=rho, xpre=xpre, xpost=xpost, wv=wv)
        st = dict(st, hist=hist)
        # astrocytes (De Pitta 2009, Li-Rinzel + IP3 metabolism), rates per second at 1 ms
        if FL["glia"]:
            dom_spk = jax.ops.segment_sum(s * in_dom, dom_idx, net.n_astro)
            st = dict(st, astro=astro_step(st["astro"], dom_spk))
        # energy: per-cell ATP from glucose and blood flow, spent on synaptic drive and spikes
        if FL["energy"]:
            Ca = st["astro"][1]
            cbf = jnp.where(in_dom, 1.0 + 0.5 * Ca[dom_idx] / (Ca[dom_idx] + 0.5), 1.0)
            g_exc = G[:n_slot].reshape(-1, PR.N_REC)[:, :2].sum(1)
            g_exc = jax.ops.segment_sum(g_exc, jnp.asarray(_slot_owner(net)), n)
            Ek = ENERGY
            datp = (Ek["k_sup"] * glucose * cbf * (1.05 - atp) - Ek["k_rest"] - Ek["k_syn"] * g_exc) * 1e-3 \
                - Ek["k_spike"] * s
            atp = jnp.clip(atp + datp, 0.0, 1.05)
            nad = st["nad"] + (jnp.mean(atp) - st["nad"]) / Ek["nad_tau"]
            st = dict(st, atp=atp, nad=nad)
        # volume transmission: dM/dt = -M/tau + 0.05 * spikes of the source nucleus
        src_spk = jax.ops.segment_sum(s * (nm_src >= 0), jnp.maximum(nm_src, 0), 4)
        st = dict(st, nm=nm * nm_k + 0.05 * src_spk, t=t + 1)
        # probe: mean excitatory and inhibitory synaptic conductance on the L2/3 PCs (all compartments)
        G3 = G[o3:o2].reshape(n3, 3, PR.N_REC)
        # prediction error of the S and T L2/3 cells: basal (evidence) vs tuft (expectation) depolarisation,
        # weighted by the astrocytic precision of each cell's domain (design §6.2); a readout, never a loop
        dep_b = jnp.maximum(st["Vb"][:n_st] - PR.L23["EL"], 0.0)
        dep_a = jnp.maximum(st["Va"][:n_st] - PR.L23["EL"], 0.0)
        prec = 1.0 + 0.5 * jnp.where(in_dom[:n_st], st["astro"][3][dom_idx[:n_st]], 0.0)
        F = 0.5 * jnp.mean(prec * (dep_b - dep_a) ** 2)
        probe = jnp.concatenate([jnp.stack([G3[..., :2].sum((1, 2)).mean(), G3[..., 2:].sum((1, 2)).mean()]),
                                 st["nm"], F[None]])
        if record:   # per-ms soma voltage and peak tuft voltage of the L2/3 cells (cell-level gates)
            return st, (s, probe, jnp.stack([st["V"][:n3], va_max]))
        return st, (s, probe)

    def window(st, ext, glucose, gclamp=None, gbus=None):
        W = ext.shape[0]
        if gclamp is None:
            gclamp = jnp.zeros((W, n3), f)
        if gbus is None:
            gbus = jnp.zeros((W, n_slot + 1), f)
        return lax.scan(ms_step, st, (ext, jnp.full((W,), glucose, f), gclamp, gbus))

    if max_w is None:
        return jax.jit(window)

    def window_n(st, ext, glucose, gbus, n_ms):
        """The first ``n_ms`` <= max_w ms of padded [max_w, ...] inputs, with a dynamic trip count: one compiled
        loop for every window size, so W = 1 and W = 4 execute identical machine code (gate G6b)."""
        g = jnp.asarray(glucose, f)
        z = jnp.zeros(n3, f)
        out_s = jnp.zeros((max_w, n), f)
        out_p = jnp.zeros((max_w, PROBE_N), f)

        def body(i, carry):
            st, out_s, out_p = carry
            st, outs = ms_step(st, (ext[i], g, z, gbus[i]))
            return st, out_s.at[i].set(outs[0]), out_p.at[i].set(outs[1])

        return lax.fori_loop(0, n_ms, body, (st, out_s, out_p))

    return jax.jit(window_n)


_OWNER_CACHE: dict = {}


def _slot_owner(net: Net) -> np.ndarray:
    """For each compartment block of 5 receptor slots, the cell that owns it."""
    key = net.digest
    if key not in _OWNER_CACHE:
        ncomp = np.where(np.arange(net.n) < net.n3, 3, np.where(np.arange(net.n) < net.n3 + net.n2, 2, 1))
        _OWNER_CACHE[key] = np.repeat(np.arange(net.n), ncomp).astype(np.int32)
    return _OWNER_CACHE[key]


def population_rates(net: Net, spikes: np.ndarray, ms: float) -> dict[str, float]:
    """Mean firing rate (Hz) per population from a [T, N] spike raster covering ``ms`` milliseconds."""
    total = np.asarray(spikes).sum(0)
    return {p.name: float(total[net.idx(p.name)].mean() * 1000.0 / ms) for p in net.pops}


QUIET_MS = 20_000.0
QUIET_STEP_MS = 20.0
POST_TAU_MS = 2_000.0


def make_quiet(net: Net):
    """The quiet interval Q (design §5.6): 20 s of slow-only brain time between turns (the player reading and
    typing). Fast state (membranes, conductances, STP, eligibility) is frozen; astrocytes, ATP, NAD and the
    Graupner-Brunel drift integrate in 20-ms steps against an activity profile that decays from the turn's rates to
    the resting rates with tau 2 s. Returns a jitted ``quiet(state, glucose, rate_turn[N], rate_rest[N])``."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    steps = int(round(QUIET_MS / QUIET_STEP_MS))
    assert steps * QUIET_STEP_MS == QUIET_MS == 1000 * QUIET_STEP_MS, "Q must be 1000 steps of 20 ms"
    p = {k: jnp.asarray(v) for k, v in net.point.items()}
    astro_of = p["astro_of"]
    in_dom = astro_of >= 0
    dom_idx = jnp.maximum(astro_of, 0)
    E = ENERGY
    dt_s = QUIET_STEP_MS * 1e-3

    def quiet(st, glucose, rate_turn, rate_rest):
        def body(k, carry):
            astro, atp, rho, nad = carry
            r = rate_rest + (rate_turn - rate_rest) * jnp.exp(-k * QUIET_STEP_MS / POST_TAU_MS)   # Hz
            dom = jax.ops.segment_sum(r * 1e-3 * in_dom, dom_idx, net.n_astro)                     # spikes/ms
            astro = astro_step(astro, dom, QUIET_STEP_MS)
            Ca = astro[1]
            cbf = jnp.where(in_dom, 1.0 + 0.5 * Ca[dom_idx] / (Ca[dom_idx] + 0.5), 1.0)
            atp = jnp.clip(atp + (E["k_sup"] * glucose * cbf * (1.05 - atp) - E["k_rest"]
                                  - E["k_spike"] * r) * dt_s, 0.0, 1.05)
            rho = jnp.clip(rho + QUIET_STEP_MS * gb_drho(rho, 0.0, nad), 0.0, 1.0)    # calcium-free drift only
            nad = nad + (jnp.mean(atp) - nad) * QUIET_STEP_MS / E["nad_tau"]
            return astro, atp, rho, nad

        astro, atp, rho, nad = lax.fori_loop(0, steps, body, (st["astro"], st["atp"], st["rho"], st["nad"]))
        return dict(st, astro=astro, atp=atp, rho=rho, nad=nad)

    return jax.jit(quiet)

