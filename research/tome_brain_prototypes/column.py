"""Prototype JAX cortical-column partition for timing: 3 columns, each with 9-compartment pyramidal cells
(L2/3 IT 256, L5 ET 48, L5 IT 48) and single-compartment AdEx cells (L4 128, L6 CT 64, PV 32, SST 24, VIP 16,
LAMP5 16). One global tick = 0.5 ms; membranes sub-step at dt = 0.05 ms (10 substeps per tick); synaptic
conductances update once per tick from the previous tick's spikes (intra-partition delay >= 0.5 ms).
Short-term plasticity (Tsodyks-Markram) on every presynaptic cell; GB calcium traces per PC; astrocytes."""
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23cell as L

N_PC = 256 + 48 + 48
N_AD = 128 + 64 + 32 + 24 + 16 + 16
N = N_PC + N_AD
COLS = 3
SUB = 10
DT = 0.05
TICK = SUB * DT

# AdEx (Brette & Gerstner 2005), per-cell params (generic fast/slow mix)
C_ad, gL_ad, EL_ad, VT_ad, DT_ad, a_ad, tau_w, b_ad, Vr_ad = 150.0, 10.0, -65.0, -50.0, 2.0, 2.0, 100.0, 40.0, -58.0

def init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    pc = jax.tree.map(lambda x: jnp.broadcast_to(x, (COLS,) + x.shape), L.init(N_PC))
    W = jax.random.uniform(k1, (COLS, N, N)) * (jax.random.uniform(k2, (COLS, N, N)) < 0.1) * 0.002
    return dict(pc=pc, v_ad=jnp.full((COLS, N_AD), EL_ad), w_ad=jnp.zeros((COLS, N_AD)),
                g_e=jnp.zeros((COLS, N_PC, L.NC)), g_i=jnp.zeros((COLS, N_PC, L.NC)),
                g_ad=jnp.zeros((COLS, N_AD)), x=jnp.ones((COLS, N)), u=jnp.full((COLS, N), 0.35),
                ca=jnp.zeros((COLS, N_PC)), rho=jnp.full((COLS, N_PC), 0.5), spk=jnp.zeros((COLS, N)),
                astro=jnp.zeros((COLS, 4, 32)), W=W)

def tick(st, ext):
    """ext: [COLS, N] external (bus) conductance drive for this tick."""
    spk = st["spk"]
    # Tsodyks-Markram STP (exact exponential relaxation over one tick, then release on spikes)
    U0, tau_d, tau_f = 0.35, 500.0, 667.0
    x = 1.0 - (1.0 - st["x"]) * jnp.exp(-TICK / tau_d)
    u = U0 + (st["u"] - U0) * jnp.exp(-TICK / tau_f)
    u = u + U0 * (1 - u) * spk
    rel = u * x * spk
    x = x - rel
    drive = jnp.einsum("cij,cj->ci", st["W"], rel) + ext  # [COLS, N]
    g_e = st["g_e"] * jnp.exp(-TICK / 2.0)
    g_e = g_e.at[:, :, 1:7].add(drive[:, :N_PC, None] / 6.0)
    g_i = st["g_i"] * jnp.exp(-TICK / 8.0)
    g_ad = st["g_ad"] * jnp.exp(-TICK / 3.0) + drive[:, N_PC:]

    def sub(carry, _):
        pc, v, w, fired_pc, fired_ad = carry
        pc = jax.vmap(lambda p, ge, gi: L.step(p, ge, gi, jnp.zeros(N_PC), DT))(pc, g_e, g_i)
        fired_pc = fired_pc | (pc["V"][..., 0] > 0.0)
        I = -gL_ad * (v - EL_ad) + gL_ad * DT_ad * jnp.exp((jnp.minimum(v, -30.0) - VT_ad) / DT_ad) - w \
            - g_ad * 50.0 * (v - 0.0) / 50.0
        v = v + DT * I / C_ad
        w = w + DT * (a_ad * (v - EL_ad) - w) / tau_w
        f = v > -30.0
        v = jnp.where(f, Vr_ad, v); w = jnp.where(f, w + b_ad, w)
        return (pc, v, w, fired_pc, fired_ad | f), None

    z = jnp.zeros((COLS, N_PC), bool), jnp.zeros((COLS, N_AD), bool)
    (pc, v, w, fpc, fad), _ = jax.lax.scan(sub, (st["pc"], st["v_ad"], st["w_ad"]) + z, None, length=SUB)
    spk_new = jnp.concatenate([fpc, fad], 1).astype(jnp.float32)
    # Graupner-Brunel: calcium from post spikes (proxy), bistable rho
    ca = st["ca"] * jnp.exp(-TICK / 20.0) + 1.0 * fpc
    rho = st["rho"]
    drho = (-rho * (1 - rho) * (0.5 - rho) + 0.35 * (1 - rho) * (ca > 1.3) - 0.15 * rho * (ca > 1.0)) / 150000.0
    rho = jnp.clip(rho + TICK * drho, 0, 1)
    return dict(st, pc=pc, v_ad=v, w_ad=w, g_e=g_e, g_i=g_i, g_ad=g_ad, x=x, u=u, ca=ca, rho=rho, spk=spk_new), spk_new

@jax.jit
def epoch(st, ext_ticks):
    """Advance one min-delay epoch: ext_ticks [ticks, COLS, N]."""
    return jax.lax.scan(tick, st, ext_ticks)
