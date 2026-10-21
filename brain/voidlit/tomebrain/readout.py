"""Reading the Tome's mind: needs, drive, desperation, affect, and its own check modifiers (design §8).

Readouts are computed from what the network did this turn: population rates, ATP, neuromodulator concentrations
and prediction error. They only *name* the dynamics; behaviour comes from the network itself. They are hidden:
they feed the rules' gates, the voice's register (later) and developer telemetry, never a frame.

    survival     ATP of the schemer and insula columns (felt as orexin hunger firing)
    affiliation  the amygdala's bond assembly against its threat and contempt assemblies
    certainty    1 - normalised prediction error of the S and T L2/3 cells
    competence   the ventral-striatum value trend

Drive is Keramati-Gutkin, D(H) = (sum_i |1 - h_i|^2.4)^(1/1.8). Its reduction is the intrinsic reward r_t, which
enters the VTA as a lateral-hypothalamic current at the next Phase A: the one place an AURA-X equation acts on the
network directly, and it is declared as such.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

N_EXP, M_EXP = 2.4, 1.8
D_MAX = 4 ** (1 / M_EXP)
EMA_TURNS = 50
NEED_SMOOTH = 0.35      # needs are states, not flashes: each turn moves them 35% of the way
NEEDS = ("survival", "affiliation", "certainty", "competence")


def drive(h: dict[str, float]) -> float:
    return sum(abs(1.0 - h[k]) ** N_EXP for k in NEEDS) ** (1.0 / M_EXP)


def desperation(d: float) -> float:
    return min(1.0, max(0.0, (d - 0.6) / 0.9))


@dataclass
class TurnSignals:
    """What the network did this turn (window means)."""

    atp_pi: float          # mean ATP of the P and I columns
    rates: dict[str, float]  # population rates (Hz): VTA, LC, DRN, NBM, ORX, PVN, plus bla.* and cea, vs
    nm: list[float]        # DA, NE, 5-HT, ACh concentrations (turn mean)
    F: float               # mean precision-weighted prediction error


@dataclass
class Running:
    mean: float = 0.0
    var: float = 1.0
    n: int = 0

    def update(self, x: float) -> None:
        self.n += 1
        k = 1.0 / min(self.n, EMA_TURNS)
        d = x - self.mean
        self.mean += k * d
        self.var += k * (d * d - self.var)

    def z(self, x: float) -> float:
        return (x - self.mean) / math.sqrt(max(self.var, 1e-6)) if self.n > 2 else 0.0


@dataclass
class Mind:
    needs: dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in NEEDS})
    drive: float = 0.0
    reward: float = 0.0
    desperation: float = 0.0
    contempt: float = 0.0
    pad: dict[str, float] = field(default_factory=lambda: {"P": 0.0, "A": 0.0, "D": 0.0})
    affect: dict[str, float] = field(default_factory=lambda: {"bond": 0.0, "hunger": 0.0, "fear": 0.0})
    mods: dict[str, int] = field(default_factory=lambda: {"grit": 2, "guile": 2, "insight": 2, "resolve": 3})
    stats: dict[str, Running] = field(default_factory=dict)
    vs_prev: float = 0.0

    def _z(self, key: str, x: float) -> float:
        r = self.stats.setdefault(key, Running())
        z = r.z(x)
        r.update(x)
        return z

    def update(self, sig: TurnSignals, tom_confidence: float = 0.4, io_error: float = 0.5) -> "Mind":
        """Fold one turn into the readouts. ``tom_confidence`` and ``io_error`` come from parts built later
        (Theory of Mind, the cerebellum); until then they sit at neutral values."""
        r = sig.rates
        z = {k: self._z(k, v) for k, v in (("LC", r["LC"]), ("ORX", r["ORX"]), ("DA", sig.nm[0]),
                                            ("5HT", sig.nm[2]), ("ACh", sig.nm[3]), ("CeA", r["cea"]),
                                            ("F", sig.F), ("ATP", sig.atp_pi), ("VS", r["vs"]),
                                            ("contempt", r["bla.contempt"]), ("bond", r["bla.bond"]))}
        f_rest = self.stats["F"].mean or sig.F or 1.0
        bond = r["bla.bond"]
        rival = r["bla.threat"] + r["bla.contempt"]
        h = {
            "survival": min(1.0, max(0.0, (sig.atp_pi - 0.55) / 0.44)),
            "affiliation": 0.5 + 0.5 * math.tanh((bond - rival) / (bond + rival + 1.0)),
            "certainty": math.exp(-max(0.0, sig.F / max(f_rest, 1e-6) - 1.0)),
            "competence": 0.5 + 0.5 * math.tanh(r["vs"] - self.vs_prev),
        }
        self.vs_prev = r["vs"]
        if self.stats["F"].n > 1:
            h = {k: self.needs[k] + NEED_SMOOTH * (h[k] - self.needs[k]) for k in NEEDS}
        d = drive(h)
        self.reward = self.drive - d if self.stats["F"].n > 1 else 0.0
        self.needs, self.drive = h, d
        self.desperation = desperation(d)
        self.contempt = 0.5 + 0.5 * math.tanh(z["contempt"])
        self.pad = {
            "P": math.tanh(1.2 * z["DA"] + 0.6 * z["5HT"] - 0.8 * z["CeA"] - 0.8 * z["ORX"]),
            "A": math.tanh(z["LC"] + 0.5 * z["ACh"] + 0.5 * z["CeA"]),
            "D": math.tanh(0.6 * z["ATP"] - 0.8 * z["CeA"] + 0.4 * z["contempt"]),
        }
        # the voice's register axes (design §11.6): bond, hunger = z(ORX), fear = z(CeA), each -1..1
        self.affect = {"bond": math.tanh(z["bond"]), "hunger": math.tanh(z["ORX"]), "fear": math.tanh(z["CeA"])}
        # the Tome's own check modifiers (design §8.6), each in [-2, +6]
        clip = lambda x: int(max(-2, min(6, round(x))))  # noqa: E731
        self.mods = {
            "grit": clip(-2 + 8 * min(1.0, max(0.0, (sig.atp_pi - 0.4) / 0.6)) + (z["LC"] > 1)),
            "guile": clip(-1 + 6 * tom_confidence + 2 * (z["DA"] > 0) - 2 * (z["ORX"] > 1.5)),
            "insight": clip(5 - 6 * io_error),
            "resolve": clip(4 - 3 * d / D_MAX + 2 * math.tanh(z["5HT"])),
        }
        return self

    def telemetry(self) -> dict:
        d = asdict(self)
        d.pop("stats")
        return d

    def state_dict(self) -> dict:
        return {"needs": dict(self.needs), "drive": self.drive, "reward": self.reward,
                "desperation": self.desperation, "contempt": self.contempt, "pad": dict(self.pad),
                "affect": dict(self.affect),
                "mods": dict(self.mods), "vs_prev": self.vs_prev,
                "stats": {k: [v.mean, v.var, v.n] for k, v in self.stats.items()}}

    def load_state_dict(self, d: dict) -> None:
        self.needs = {k: float(v) for k, v in d["needs"].items()}
        self.drive, self.reward = float(d["drive"]), float(d["reward"])
        self.desperation, self.contempt = float(d["desperation"]), float(d["contempt"])
        self.pad = {k: float(v) for k, v in d["pad"].items()}
        self.affect = {k: float(v) for k, v in d.get("affect", self.affect).items()}
        self.mods = {k: int(v) for k, v in d["mods"].items()}
        self.vs_prev = float(d["vs_prev"])
        self.stats = {k: Running(float(v[0]), float(v[1]), int(v[2])) for k, v in d["stats"].items()}


def population_rates(net, counts: np.ndarray, ms: float) -> dict[str, float]:
    return {p.name: float(counts[net.idx(p.name)].mean() * 1000.0 / max(ms, 1.0)) for p in net.pops}


def atp_pi(net, atp: np.ndarray) -> float:
    cells = np.concatenate([np.arange(net.idx(f"{c}.{pop}").start, net.idx(f"{c}.{pop}").stop)
                            for c in ("P", "I") for pop in ("L23", "L5ET", "L5IT", "L4", "L6CT")])
    return float(np.asarray(atp)[cells].mean())


__all__ = ["Mind", "TurnSignals", "drive", "desperation", "population_rates", "atp_pi", "NEEDS", "D_MAX"]
