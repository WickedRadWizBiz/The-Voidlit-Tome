"""G5 rate calibration (design §7.5): make the circuit alive and balanced before any behavioural claim.

Runs the JAX partition in two input conditions, **rest** (thalamic channels at a quiet baseline) and **busy** (a
scripted trace of changing channel drive), and fits each population's tonic current and the inhibitory gains onto
the L2/3 PCs until every population sits in its physiological band and the PC conductance ratio is E:I 1:2-1:4.

    python -m voidlit.tomebrain.calib.calib_rates            # fit and write connectome_calibrated.toml

The result is versioned and hashed with the connectome; ``network.build`` loads it by default.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ..jaxpart import network as NW
from ..jaxpart import params as PR

CALIBRATION_FILE = NW.CALIBRATION_FILE
# The file's last section: the settled network's resting rates at Lumen 0..6, which the brain's quiet interval relaxes
# toward (``brain.measure_rest_by_lumen`` and ``brain.write_rest_by_lumen``).
REST_BY_LUMEN = "rates_rest_by_lumen"
NOTE_BY_LUMEN = "# Resting rates at Lumen 0..6 (voidlit.tomebrain.brain.measure_rest_by_lumen). Regenerate; keep last."

# (lo, hi) Hz in both conditions. Populations the design does not list get neighbouring bands.
_BANDS = {
    "L23": (0.5, 3.0), "L4": (2.0, 6.0), "L5ET": (3.0, 10.0), "L5IT": (1.0, 6.0), "L6CT": (1.0, 6.0),
    "PV23": (5.0, 20.0), "PV4": (5.0, 20.0), "PV5": (5.0, 20.0), "PV6": (5.0, 20.0),
    "SST23": (2.0, 8.0), "SST5": (2.0, 8.0), "SST6": (2.0, 8.0), "VIP": (2.0, 6.0), "NGF": (2.0, 8.0),
    "TC": (5.0, 20.0), "MTX": (5.0, 20.0), "TRN": (5.0, 25.0), "PVT": (2.0, 10.0),
    "DG": (0.05, 1.0), "CA3": (0.5, 4.0), "CA1": (0.5, 4.0), "CA3.PV": (5.0, 20.0), "CA1.PV": (5.0, 20.0),
    "OLM": (2.0, 8.0), "VTA": (3.0, 6.0), "RMTg": (2.0, 8.0), "LC": (1.0, 3.0), "DRN": (1.0, 3.0),
    "NBM": (1.0, 3.0), "ORX": (0.5, 3.0), "PVN": (1.0, 4.0),
}
EI_BAND = (2.0, 4.0)        # inhibitory : excitatory conductance on L2/3 PCs
CV_BAND = (0.8, 1.2)        # L2/3 interspike-interval variability (CV2)
REST_DRIVE = 0.2            # thalamic channel drive at rest (fraction of 220 pA)
CHANNEL_PA = 220.0


def band(pop: str) -> tuple[float, float]:
    key = pop.split(".", 1)[1] if pop[:2] in ("S.", "T.", "P.", "I.") else pop
    return _BANDS[key]


def thalamic_drive(net: NW.Net, ms: int, busy: bool, seed: int = 0) -> np.ndarray:
    """External current [ms, N]: two TC core cells per channel, 220 pA x f. Busy changes f every 250 ms."""
    ext = np.zeros((ms, net.n), np.float32)
    rng = np.random.default_rng(1000 + seed)
    for c in PR.COLUMNS:
        sl = net.idx(f"{c}.TC")
        n_ch = (sl.stop - sl.start) // 2
        for t0 in range(0, ms, 250):
            f = rng.uniform(0.0, 1.0, n_ch) if busy else np.full(n_ch, REST_DRIVE)
            ext[t0:t0 + 250, sl] = np.repeat(f, 2)[None, :] * CHANNEL_PA
    return ext


def isi_cv(raster: np.ndarray) -> float:
    """Local ISI variability CV2 (Holt et al. 1996): mean of 2|I(i+1) - I(i)| / (I(i+1) + I(i)).

    It is 1 for a Poisson train at any rate and, unlike the global CV, is not biased low by short windows or by
    rate differences between cells (a 2.5-s window gives a Poisson train at 1 Hz a global CV of only 0.69)."""
    vals: list[float] = []
    for col in raster.T:
        t = np.flatnonzero(col)
        if len(t) >= 3:
            isi = np.diff(t).astype(float)
            vals.extend(2.0 * np.abs(np.diff(isi)) / (isi[1:] + isi[:-1]))
    return float(np.mean(vals)) if vals else float("nan")


class Probe:
    """Builds a network and its jitted window once; runs conditions with a given tonic-current vector."""

    def __init__(self, gains: dict | None = None, seed: int = 0, flags: dict | None = None, W: int = 4):
        import jax

        self.net = NW.build(seed=seed, gains=gains, calibrated=False)
        self.window = NW.make_window(self.net, flags=flags)
        self.key = jax.random.PRNGKey(seed)
        self.W = W

    def run(self, mu: np.ndarray, busy: bool, ms: int = 2500, warm: int = 500, glucose: float = 1.0,
            sd_scale: float = 1.0):
        import jax.numpy as jnp

        st = NW.init_state(self.net, self.key)
        st["mu"] = jnp.asarray(mu, jnp.float32)
        st["sd"] = jnp.asarray(self.net.point["sd"] * sd_scale, jnp.float32)
        ext = thalamic_drive(self.net, warm + ms, busy)
        spikes, probes = [], []
        for k in range(0, warm + ms, self.W):
            st, (s, pr) = self.window(st, jnp.asarray(ext[k:k + self.W]), glucose)
            if k >= warm:
                spikes.append(np.asarray(s))
                probes.append(np.asarray(pr))
        R = np.concatenate(spikes)
        P = np.concatenate(probes)
        rates = NW.population_rates(self.net, R, ms)
        ei = float(P[:, 1].mean() / max(P[:, 0].mean(), 1e-6))
        l23 = np.concatenate([R[:, self.net.idx(f"{c}.L23")] for c in PR.COLUMNS], axis=1)
        return dict(rates=rates, ei=ei, cv=isi_cv(l23), overflow=int(st["overflow"]),
                    peak=int(R.sum(1).max()), state=st)


def in_band(rates: dict) -> dict[str, bool]:
    return {p: band(p)[0] <= r <= band(p)[1] for p, r in rates.items()}


def calibrate(rounds: int = 40, verbose: bool = True, seed: int = 0, warm_start: bool = True) -> dict:
    net0 = NW.build(seed=seed, calibrated=False)
    names = [p.name for p in net0.pops]
    mu = {p.name: p.mu for p in net0.pops}
    inh_gain, sd_scale = 1.0, 1.0
    if warm_start and CALIBRATION_FILE.exists():
        cal = NW.load_calibration()
        mu.update(cal["mu"])
        inh_gain, sd_scale = cal["inh_gain"], cal["sd_scale"]
    step = {k: 30.0 for k in names}
    last_sign = {k: 0 for k in names}
    inh_idx = NW.inhibitory_onto_l23()
    probe = Probe(gains={k: inh_gain for k in inh_idx}, seed=seed)
    best = None
    for r in range(rounds):
        t0 = time.time()
        mu_vec = _mu_vector(probe.net, mu)
        rest, busy = probe.run(mu_vec, False, sd_scale=sd_scale), probe.run(mu_vec, True, sd_scale=sd_scale)
        ok_r, ok_b = in_band(rest["rates"]), in_band(busy["rates"])
        n_bad = sum(not v for v in ok_r.values()) + sum(not v for v in ok_b.values())
        ei_ok = EI_BAND[0] <= rest["ei"] <= EI_BAND[1]
        if verbose:
            bad = sorted({p for p in names if not (ok_r[p] and ok_b[p])})
            print(f"round {r:2d}: out of band {n_bad:3d} | E:I 1:{rest['ei']:.2f} | CV {rest['cv']:.2f} "
                  f"(bg x{sd_scale:.2f}) | "
                  f"peak {busy['peak']} | {time.time() - t0:.1f}s | {bad[:12]}", flush=True)
        cv_ok = CV_BAND[0] <= rest["cv"] <= CV_BAND[1]
        score = n_bad + (0 if ei_ok else 5) + (0 if cv_ok else 3)
        if best is None or score < best[0]:
            best = (score, dict(mu), inh_gain, rest, busy, sd_scale)
        if n_bad == 0 and ei_ok and cv_ok:
            break
        for p in names:
            lo, hi = band(p)
            target = math.sqrt(lo * hi)
            err = math.log((target + 0.3) / (rest["rates"][p] + 0.3))
            if busy["rates"][p] > hi:
                err += math.log((hi + 0.3) / (busy["rates"][p] + 0.3))
            elif busy["rates"][p] < lo:
                err += 0.5 * math.log((lo + 0.3) / (busy["rates"][p] + 0.3))
            err = max(-1.5, min(1.5, err))
            sign = (err > 0.05) - (err < -0.05)
            if sign and last_sign[p] and sign != last_sign[p]:
                step[p] = max(2.0, step[p] * 0.5)
            elif sign:
                step[p] = min(60.0, step[p] * 1.15)
            last_sign[p] = sign or last_sign[p]
            mu[p] += step[p] * err
        # Irregular firing needs a fluctuation-driven regime. A 2,000-cell network lacks the synaptic bombardment
        # of a whole cortex, so the background fluctuation amplitude is fitted too (a declared calibration).
        if rest["cv"] < CV_BAND[0]:
            sd_scale = min(5.0, sd_scale * 1.2)
        elif rest["cv"] > CV_BAND[1]:
            sd_scale = max(0.5, sd_scale / 1.2)
        target_ei = math.sqrt(EI_BAND[0] * EI_BAND[1])
        if abs(target_ei - rest["ei"]) / target_ei > 0.08:
            inh_gain *= max(0.7, min(1.4, 1.0 + 0.25 * (target_ei - rest["ei"]) / target_ei))
            probe = Probe(gains={k: inh_gain for k in inh_idx}, seed=seed)
    score, mu, inh_gain, rest, busy, sd_scale = best
    return dict(mu=mu, inh_gain=inh_gain, inh_idx=inh_idx, rest=rest, busy=busy, score=score, sd_scale=sd_scale)


def _mu_vector(net: NW.Net, mu: dict) -> np.ndarray:
    v = np.zeros(net.n, np.float32)
    for p in net.pops:
        v[net.idx(p.name)] = mu[p.name]
    return v


def rest_by_lumen_section(path: Path) -> str:
    """The file's resting-rates-by-Lumen section (its note, heading and table), or "" when it has none."""
    text = path.read_text() if path.exists() else ""
    head = f"[{REST_BY_LUMEN}]"
    if head not in text:
        return ""
    start = text.index(head)
    if text[:start].rstrip("\n").endswith(NOTE_BY_LUMEN):
        start = text[:start].rstrip("\n").rindex(NOTE_BY_LUMEN)
    return text[start:]


def write(result: dict, path: Path = CALIBRATION_FILE) -> None:
    """Write a calibration, keeping the file's resting rates by Lumen (measured on the network this calibration
    replaces: ``main`` measures them again). Without them the brain would rest a starving network fed."""
    keep = rest_by_lumen_section(path)
    lines = ["# G5 rate calibration (voidlit.tomebrain.calib.calib_rates). Regenerate; do not edit by hand.",
             f"score = {result['score']}", f"inhibitory_gain_onto_l23 = {result['inh_gain']:.4f}",
             f"background_sd_scale = {result['sd_scale']:.4f}", "", "[mu]"]
    lines += [f'"{k}" = {v:.2f}' for k, v in sorted(result["mu"].items())]
    lines += ["", "[rates_rest]"] + [f'"{k}" = {v:.3f}' for k, v in sorted(result["rest"]["rates"].items())]
    lines += ["", "[rates_busy]"] + [f'"{k}" = {v:.3f}' for k, v in sorted(result["busy"]["rates"].items())]
    lines += ["", "[measures]", f"ei_rest = {result['rest']['ei']:.3f}", f"cv_rest = {result['rest']['cv']:.3f}",
              f"peak_spikes_per_ms = {result['busy']['peak']}", f"overflow = {result['busy']['overflow']}"]
    path.write_text("\n".join(lines) + "\n" + ("\n" + keep if keep else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--out", type=Path, default=CALIBRATION_FILE)
    ap.add_argument("--fresh", action="store_true", help="start from params.py, not the current calibration")
    ap.add_argument("--keep-rest-by-lumen", action="store_true",
                    help="keep the old resting rates by Lumen instead of measuring them on the new calibration")
    args = ap.parse_args()
    res = calibrate(args.rounds, warm_start=not args.fresh)
    write(res, args.out)
    print(f"wrote {args.out} (score {res['score']})")
    if args.out.resolve() == CALIBRATION_FILE.resolve() and not args.keep_rest_by_lumen:
        from ..brain import measure_rest_by_lumen, write_rest_by_lumen     # (measures the network this file builds)
        write_rest_by_lumen(measure_rest_by_lumen(), args.out)
        print(f"measured the resting rates by Lumen into {args.out}")


if __name__ == "__main__":
    main()
