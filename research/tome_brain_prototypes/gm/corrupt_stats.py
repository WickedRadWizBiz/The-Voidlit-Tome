"""Corruption bursts from cell dynamics, then the invisible check, refractory resistance and cap.

Puppet population = 16 L5 ET cells per NPC (4-compartment model with a monotone BAC Ca spike: L5's classic Ca spike,
not the human L2/3 dCaAP).  Basal input = the agenda's need for a puppet voice (fixed here); tuft input = desperation,
relayed by matrix thalamus.  A cell 'bursts' if it fires >= 2 spikes within 15 ms.  A corruption ATTEMPT happens when
>= 6 of 16 cells burst in the 250 ms turn window.  Then: Tome d20 + corruption mod vs NPC DC = 10 + 2*Resolve + 12*resist.
"""
import sys, random
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L

L.DCA_SIG = 1e9            # monotone Ca spike (L5 BAC)
L.DCA_GBAR = 4.0
N_CELL, TURNS, MS, DT = 16, 300, 250.0, 0.1
STEPS = int(MS / DT)


def bursts_per_turn(desperation, seed, basal_hz=150.0, g_b=1.0, tuft_hz0=50.0, tuft_hz1=200.0, g_t=1.2):
    n = N_CELL * TURNS
    key = jax.random.PRNGKey(seed)
    # per-turn background jitter of the drives (the game state differs from turn to turn)
    k0, key = jax.random.split(key)
    jit_t = jnp.repeat(jnp.exp(0.35 * jax.random.normal(k0, (TURNS,))), N_CELL)
    rb = basal_hz * DT / 1000.0
    rt = (tuft_hz0 + tuft_hz1 * desperation) * DT / 1000.0 * jit_t

    @jax.jit
    def run(key):
        s = L.init(n)
        def body(c, k):
            s, key, last, isi_ok = c
            key, a, b = jax.random.split(key, 3)
            wb = (jax.random.uniform(a, (n,)) < rb) * g_b
            wt = (jax.random.uniform(b, (n,)) < rt) * g_t
            wa = jnp.zeros((n, 4)).at[:, 1].set(wb).at[:, 3].set(wt)
            s, sp = L.step(s, wa, 0.6 * wa, jnp.zeros((n, 4)), jnp.zeros(n), DT)
            t = k * DT
            burst = sp & ((t - last) < 15.0)
            return (s, key, jnp.where(sp, t, last), isi_ok | burst), sp
        (s, key, last, burst), sp = jax.lax.scan(body, (s, key, jnp.full(n, -1e9), jnp.zeros(n, bool)), jnp.arange(STEPS))
        return burst, sp.sum(0)
    burst, count = run(key)
    burst = np.asarray(burst).reshape(TURNS, N_CELL)
    rate = float(np.asarray(count).mean()) / (MS / 1000.0)
    return burst.sum(1), rate


def checks(burst_counts, desperation, resolve, seed, cap=0.35):
    rng = random.Random(seed)
    resist, out, last_t, gaps = 0.0, [], None, []
    for t, nb in enumerate(burst_counts):
        resist *= 0.6                                     # refractory recovery (tau ~ 2 turns)
        attempt = nb >= 6 and not (last_t is not None and t - last_t <= 1)   # absolute refractory: 1 turn
        ok, s = False, 0.0
        if attempt:
            mod = round(2 + 6 * desperation)
            dc = 10 + 2 * resolve + round(12 * resist)
            r = rng.randint(1, 20)
            ok = (r + mod >= dc and r != 1) or r == 20
            if ok:
                s = min(cap, 0.10 + 0.02 * (r + mod - dc) + 0.20 * desperation)
                resist = min(1.0, resist + 0.8)
                if last_t is not None:
                    gaps.append(t - last_t)
                last_t = t
        out.append((attempt, ok, s))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean(), (a[a[:, 1] > 0, 2].mean() if a[:, 1].any() else 0.0), a[:, 2].max(), \
        (min(gaps) if gaps else None)


if __name__ == "__main__":
    for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        nb, rate = bursts_per_turn(d, seed=int(d * 10) + 1)
        for npc, res in (("Elara", 2), ("Marcus", 3)):
            att, succ, mean_s, max_s, gap = checks(nb, d, res, seed=7)
            print(f"desperation={d:.1f} {npc:6s}: puppet rate {rate:5.1f} Hz, attempt {att:5.1%}, corrupted turns {succ:5.1%}"
                  f" (acts as itself {1 - succ:5.1%}), mean strength {mean_s:.2f}, max {max_s:.2f} (cap 0.35), min gap {gap}",
                  flush=True)
