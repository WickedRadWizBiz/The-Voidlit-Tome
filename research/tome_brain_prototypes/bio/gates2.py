"""Gate 2/3 (honest versions), exercising the same update rules and constants the network uses (tome_jax.GB,
tome_jax.VOGELS, tome_jax.AST, cells.substep).

2a Graupner-Brunel 2012 (DP set, theta_p = 1.3): post-before-pre pairing (dt = -10 ms, 60 pairs) gives LTD at 1 Hz
   and LTP at 20 Hz (Sjostrom 2001 frequency dependence). Mutant theta_p = 1.6 (the spec's value) must fail it.
2b Vogels 2011 iSTDP on a spiking human L2/3 cell: from weak and from strong initial inhibition, the output rate
   converges to rho0 = 3 Hz. Mutant (eta = 0) must not converge.
3a De Pitta 2009 astrocyte: no Ca2+ events without synaptic glutamate; repetitive Ca2+ events (and gliotransmitter
   release) with sustained glutamate.
"""
import numpy as np
import jaxpath  # noqa
import jax
import jax.numpy as jnp

import cells
import tome_jax as TJ

GB = TJ.GB


def gb_protocol(freq_hz, dt_pair_ms, n_pairs=60, theta_p=GB["theta_p"], dt=0.5):
    period = 1000.0 / freq_hz
    T = n_pairs * period + 200.0
    steps = int(T / dt)
    t = np.arange(steps) * dt
    pre_t = np.arange(n_pairs) * period + 50.0
    post_t = pre_t + dt_pair_ms
    pre = np.zeros(steps); post = np.zeros(steps)
    pre[np.clip(np.round((pre_t + GB["D"]) / dt).astype(int), 0, steps - 1)] = 1   # presynaptic Ca delay D
    post[np.clip(np.round(post_t / dt).astype(int), 0, steps - 1)] = 1
    dec = np.exp(-dt / GB["tau_ca"])

    def body(carry, x):
        rho, cp, cq = carry
        cp = cp * dec + GB["C_pre"] * x[0]
        cq = cq * dec + GB["C_post"] * x[1]
        c = cp + cq
        drift = (-rho * (1 - rho) * (GB["rho_star"] - rho) + GB["gamma_p"] * (1 - rho) * (c >= theta_p)
                 - GB["gamma_d"] * rho * (c >= GB["theta_d"]))
        return (jnp.clip(rho + dt * drift / GB["tau"], 0, 1), cp, cq), None
    (rho, _, _), _ = jax.lax.scan(body, (jnp.float32(0.5), 0.0, 0.0), jnp.asarray(np.stack([pre, post], 1)))
    return float(rho)


E_W = 0.6
N_I = 100


def vogels_cell(w0, eta=TJ.VOGELS["eta"], seconds=180.0, seed=0):
    p = cells.build_params(["pc23h"], g_katp=0.0)
    n_e, n_i = 160, N_I
    dt = 0.25
    steps_ms = int(seconds * 1000)
    key = jax.random.PRNGKey(seed)
    tau = TJ.VOGELS["tau"]; alpha = 2 * TJ.VOGELS["rho0_hz"] * 1e-3 * tau

    def ms(carry, k):
        s, w, xpre, xpost, cnt = carry
        k1, k2 = jax.random.split(k)
        e = (jax.random.uniform(k1, (n_e,)) < 0.010).astype(jnp.float32)       # 10 Hz excitatory drive
        i = (jax.random.uniform(k2, (n_i,)) < 0.010).astype(jnp.float32)       # 10 Hz PV input
        s = dict(s, gA_b=s["gA_b"] + E_W * e.sum(), gN_b=s["gN_b"] + 0.5 * E_W * e.sum(),
                 gGA_s=s["gGA_s"] + (w * i).sum())
        spk_ms = 0.0
        for _ in range(4):
            s, spk = cells.substep(p, s, jnp.zeros((3, 1)), dt)
            spk_ms = spk_ms + spk[0]
        post = jnp.minimum(spk_ms, 1.0)
        xpre = xpre * np.exp(-1 / tau) + i
        xpost = xpost * np.exp(-1 / tau) + post
        w = jnp.clip(w + eta * (i * (xpost - alpha) + xpre * post), 0.0, TJ.VOGELS["w_max"])
        return (s, w, xpre, xpost, cnt + post), None
    s0 = cells.init_state(p)
    keys = jax.random.split(key, steps_ms)
    carry = (s0, jnp.full(n_i, w0), jnp.zeros(n_i), jnp.zeros(()), jnp.zeros(()))
    rates = []
    run = jax.jit(lambda c, ks: jax.lax.scan(ms, c, ks)[0])
    for chunk in range(int(seconds // 10)):
        c0 = carry[4]
        carry = run(carry, keys[chunk * 10000:(chunk + 1) * 10000])
        rates.append(float(carry[4] - c0) / 10.0)
    return rates, float(carry[1].mean())


def astro(glu_level, seconds=300.0, dt_ms=10.0):
    a = TJ.AST
    dts = dt_ms * 1e-3

    def body(carry, _):
        C, h, I, G, ev = carry
        m_inf = I / (I + a["d1"]); n_inf = C / (C + a["d5"])
        J_chan = a["r_C"] * m_inf ** 3 * n_inf ** 3 * h ** 3 * (a["C_T"] - (1 + a["c1"]) * C)
        J_leak = a["r_L"] * (a["C_T"] - (1 + a["c1"]) * C)
        J_pump = a["v_ER"] * C ** 2 / (C ** 2 + a["K_ER"] ** 2)
        Q2 = a["d2"] * (I + a["d1"]) / (I + a["d3"])
        h_inf = Q2 / (Q2 + C); tau_h = 1.0 / (a["a2"] * (Q2 + C))
        J_beta = a["v_beta"] * jnp.tanh(glu_level)
        J_delta = a["v_delta"] / (1 + I / a["kappa_delta"]) * C ** 2 / (C ** 2 + a["K_delta"] ** 2)
        J_3K = a["v_3K"] * C ** 4 / (C ** 4 + a["K_D"] ** 4) * I / (I + a["K_3"])
        J_5P = a["r_5P"] * I
        Cn = jnp.clip(C + dts * (J_chan + J_leak - J_pump), 0, 5)
        h = h_inf + (h - h_inf) * jnp.exp(-dts / tau_h)
        I = jnp.clip(I + dts * (J_beta + J_delta - J_3K - J_5P), 0, 10)
        up = (Cn >= a["C_theta"]) & (C < a["C_theta"])
        G = G * jnp.exp(-dts / a["tau_G"]) + a["U_A"] * (1 - G) * up
        return (Cn, h, I, G, ev + up), (Cn, G)
    (C, h, I, G, ev), (cs, gs) = jax.lax.scan(body, (0.05, 0.8, 0.05, 0.0, 0), None, length=int(seconds * 1000 / dt_ms))
    return int(ev), float(cs.max()), float(gs.max())


if __name__ == "__main__":
    import sys
    if "--vg" in sys.argv:
        for n_i, eta, wmax in ((60, 0.01, 8.0), (60, 0.02, 8.0), (100, 0.01, 6.0)):
            N_I = n_i; TJ.VOGELS["w_max"] = wmax
            for w0 in (0.2, 4.0):
                r, wm = vogels_cell(w0, eta=eta, seconds=60.0)
                print(f"n_i={n_i} eta={eta} wmax={wmax} w0={w0}: {[round(x,1) for x in r]} w={wm:.2f}", flush=True)
        raise SystemExit
    if "--drive" in sys.argv:
        for ew in (0.6, 1.0, 1.5, 2.0):
            E_W = ew
            print("E_W", ew, "uninhibited rate", vogels_cell(0.0, eta=0.0, seconds=10.0)[0])
        raise SystemExit
    r1, r20 = gb_protocol(1.0, -10.0), gb_protocol(20.0, -10.0)
    r40 = gb_protocol(40.0, -10.0)
    print(f"2a GB2012 DP set, theta_p=1.3: post-pre 1 Hz rho {r1:.3f} (LTD<0.5)  20 Hz rho {r20:.3f}  40 Hz {r40:.3f} "
          f"(LTP>0.5)  PASS={r1 < 0.5 < r20 < r40}")
    m1, m20 = gb_protocol(1.0, -10.0, theta_p=1.6), gb_protocol(20.0, -10.0, theta_p=1.6)
    print(f"   mutant theta_p=1.6: 1 Hz {m1:.3f}  20 Hz {m20:.3f}  PASS={m1 < 0.5 < m20} (must be False)")
    for w0 in (0.2, 4.0):
        rates, wm = vogels_cell(w0)
        print(f"2b Vogels iSTDP, w0={w0} nS: output rate per 10 s {[round(r, 1) for r in rates]}  final w {wm:.2f}"
              f"  PASS={abs(np.mean(rates[-3:]) - 3.0) < 1.0}")
    rates, _ = vogels_cell(0.2, eta=0.0, seconds=30.0)
    print(f"   mutant eta=0, w0=0.2: rates {[round(r, 1) for r in rates]}  (must stay far from 3 Hz)")
    for glu in (0.0, 0.3, 1.5):
        ev, cmax, gmax = astro(glu)
        print(f"3a astrocyte, glutamate drive {glu}: Ca2+ events in 300 s = {ev}, peak Ca {cmax:.2f} uM, "
              f"peak gliotransmitter {gmax:.2f}")
