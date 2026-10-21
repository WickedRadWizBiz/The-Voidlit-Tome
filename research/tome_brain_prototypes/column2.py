"""Optimised JAX partition prototype: master tick 1 ms, PC membranes sub-stepped at DT_SUB, AdEx cells at the
same substep, block-dense synapses per region updated once per tick (intra-partition delay 1 ms), TM STP,
GB calcium/rho, astrocyte domains. Whole epoch (2 ticks) per jitted call, or the whole turn per call."""
import jaxpath  # noqa
import jax, jax.numpy as jnp
import l23cell as L

N_COL_PC, N_COL_AD, COLS = 224, 232, 3
N_TH, N_HC = 304, 448
N_PC = COLS * N_COL_PC
N_AD = COLS * N_COL_AD + N_TH + N_HC
N = N_PC + N_AD
C_ad, gL_ad, EL_ad, VT_ad, DT_ad, a_ad, tau_w, b_ad, Vr_ad = 150.0, 10.0, -65.0, -50.0, 2.0, 2.0, 100.0, 40.0, -58.0

def blocks():
    """(dst_idx, src_idx) index ranges for the block-dense recurrent matrices: one block per column (its PCs +
    AdEx cells), thalamus<->all columns, hippocampus<->hippocampus+columns' L2/3."""
    cols = []
    for c in range(COLS):
        pc = list(range(c * N_COL_PC, (c + 1) * N_COL_PC))
        ad = list(range(N_PC + c * N_COL_AD, N_PC + (c + 1) * N_COL_AD))
        cols.append(pc + ad)
    th = list(range(N_PC + COLS * N_COL_AD, N_PC + COLS * N_COL_AD + N_TH))
    hc = list(range(N_PC + COLS * N_COL_AD + N_TH, N))
    return [(c, c) for c in cols] + [(th, sum(cols, [])), (sum(cols, []), th), (hc, hc + cols[0][:160])]

def init(key, dt_sub):
    ks = jax.random.split(key, 8)
    W = []
    for i, (d, s) in enumerate(blocks()):
        k1, k2 = jax.random.split(ks[i % 8])
        W.append(jax.random.uniform(k1, (len(d), len(s))) * (jax.random.uniform(k2, (len(d), len(s))) < 0.1) * 0.002)
    idx = [(jnp.array(d), jnp.array(s)) for d, s in blocks()]
    st = dict(pc=L.init(N_PC), v_ad=jnp.full(N_AD, EL_ad), w_ad=jnp.zeros(N_AD),
              g_e=jnp.zeros((N_PC, L.NC)), g_i=jnp.zeros((N_PC, L.NC)), g_ad=jnp.zeros(N_AD),
              x=jnp.ones(N), u=jnp.full(N, 0.35), ca=jnp.zeros(N_PC), rho=jnp.full(N_PC, 0.5), spk=jnp.zeros(N),
              ast=jnp.zeros((3, COLS * 28)))
    return st, W, idx

def make(dt_sub, sub):
    TICK = dt_sub * sub
    def tick(carry, ext):
        st, W, idx = carry
        spk = st["spk"]
        U0, tau_d, tau_f = 0.35, 500.0, 667.0
        x = 1.0 - (1.0 - st["x"]) * jnp.exp(-TICK / tau_d)
        u = U0 + (st["u"] - U0) * jnp.exp(-TICK / tau_f)
        u = u + U0 * (1 - u) * spk
        rel = u * x * spk
        x = x - rel
        drive = ext
        for Wb, (d, s) in zip(W, idx):
            drive = drive.at[d].add(Wb @ rel[s])
        g_e = st["g_e"] * jnp.exp(-TICK / 2.0)
        g_e = g_e.at[:, 1:7].add(drive[:N_PC, None] / 6.0)
        g_i = st["g_i"] * jnp.exp(-TICK / 8.0)
        g_ad = st["g_ad"] * jnp.exp(-TICK / 3.0) + 100.0 * drive[N_PC:]
        zero_ap = jnp.zeros(N_PC)
        def sub_(c, _):
            pc, v, w, fpc, fad = c
            pc = L.step(pc, g_e, g_i, zero_ap, dt_sub)
            fpc = fpc | (pc["V"][:, 0] > 0.0)
            vv = jnp.minimum(v, -30.0)
            I = -gL_ad * (v - EL_ad) + gL_ad * DT_ad * jnp.exp((vv - VT_ad) / DT_ad) - w - g_ad * v
            v = v + dt_sub * I / C_ad
            w = w + dt_sub * (a_ad * (v - EL_ad) - w) / tau_w
            f = v > -30.0
            return (pc, jnp.where(f, Vr_ad, v), jnp.where(f, w + b_ad, w), fpc, fad | f), None
        z = (jnp.zeros(N_PC, bool), jnp.zeros(N_AD, bool))
        (pc, v, w, fpc, fad), _ = jax.lax.scan(sub_, (st["pc"], st["v_ad"], st["w_ad"]) + z, None, length=sub)
        spk_new = jnp.concatenate([fpc, fad]).astype(jnp.float32)
        ca = st["ca"] * jnp.exp(-TICK / 22.7) + 1.08 * fpc
        rho = st["rho"]
        drho = (-rho * (1 - rho) * (0.5 - rho) + 725.0 * (1 - rho) * (ca > 1.3) - 332.0 * rho * (ca > 1.0)) / 346000.0
        rho = jnp.clip(rho + TICK * drho, 0, 1)
        st = dict(st, pc=pc, v_ad=v, w_ad=w, g_e=g_e, g_i=g_i, g_ad=g_ad, x=x, u=u, ca=ca, rho=rho, spk=spk_new)
        return (st, W, idx), spk_new
    @jax.jit
    def run(st, W, idx, ext_ticks):
        (st, _, _), spikes = jax.lax.scan(tick, (st, W, idx), ext_ticks)
        return st, spikes
    return run
