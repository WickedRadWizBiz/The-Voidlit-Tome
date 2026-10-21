"""Populations, cell kinds and the connectome of the Tome's JAX partition (design §6.1-6.7, §7.2-7.3).

Everything here is plain data (numpy-free); ``network.build`` compiles it into padded JAX tables. Units: pF, nS,
mV, ms, pA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ receptors
AMPA, NMDA, GABA_A, GABA_AS, GABA_B = range(5)
N_REC = 5
TAU_REC = (2.0, 100.0, 6.0, 20.0, 150.0)
E_REC = (0.0, 0.0, -70.0, -70.0, -90.0)
SOMA, BASAL, APICAL = 0, 1, 2

# ------------------------------------------------------------------ cell kinds
# Point cells are AdEx. Columns: C, gL, EL, VT, DT, a, b, tau_w, Vr, t_ref, I_bias, g_T (T-type, nS).
POINT_KINDS: dict[str, tuple[float, ...]] = {
    "PV":   (90.0, 9.0, -68.0, -48.0, 1.5, 0.0, 0.0, 50.0, -60.0, 1.0, 0.0, 0.0),
    "SST":  (100.0, 5.0, -65.0, -52.0, 2.0, 4.0, 40.0, 200.0, -55.0, 2.0, 0.0, 0.0),
    "VIP":  (60.0, 4.0, -65.0, -50.0, 2.0, 2.0, 20.0, 100.0, -56.0, 2.0, 0.0, 0.0),
    "NGF":  (80.0, 5.0, -70.0, -45.0, 2.0, 1.0, 10.0, 300.0, -60.0, 2.0, 0.0, 0.0),
    "RS":   (150.0, 8.0, -70.0, -50.0, 2.0, 2.0, 40.0, 120.0, -58.0, 2.0, 0.0, 0.0),
    "TC":   (200.0, 10.0, -65.0, -50.0, 2.0, 0.0, 10.0, 50.0, -55.0, 2.0, 0.0, 60.0),
    "TRN":  (150.0, 8.0, -70.0, -50.0, 2.0, 0.0, 20.0, 60.0, -58.0, 2.0, 0.0, 90.0),
    "DG":   (60.0, 5.0, -78.0, -48.0, 2.0, 2.0, 40.0, 100.0, -60.0, 2.0, 0.0, 0.0),
    "DA":   (150.0, 6.0, -60.0, -48.0, 3.0, 4.0, 50.0, 300.0, -55.0, 5.0, 140.0, 0.0),
    "PACE": (150.0, 6.0, -60.0, -48.0, 3.0, 4.0, 50.0, 400.0, -55.0, 5.0, 120.0, 0.0),
    "ORX":  (150.0, 8.0, -70.0, -50.0, 2.0, 2.0, 40.0, 120.0, -58.0, 2.0, 0.0, 0.0),
}
POINT_FIELDS = ("C", "gL", "EL", "VT", "DT", "a", "b", "tau_w", "Vr", "t_ref", "I_bias", "g_T")

# Human L2/3 pyramidal cell, three compartments (Eyal 2016): soma, basal, apical tuft.
L23 = dict(C=(72.0, 162.0, 126.0), gL=(4.1, 9.3, 7.2), g_sb=30.0, g_sa=12.0, EL=-72.0, VT=-50.0, DT=2.0,
           a=2.0, b=40.0, tau_w=150.0, Vr=-58.0, t_ref=2.0, g_kv=8.0, bap=60.0, nmda_ratio=(0.5, 0.5, 0.6))
# Gidon 2020 dCaAP in the L2/3 tuft (conductance form). u_th and tau_K set the XOR window (gate G1c).
DCAAP = dict(g_d=20.0, E_Ca=120.0, theta=-36.0, u_th=15.0, tau_K=0.30, tau_m=2.0, tau_h_fall=0.5,
             tau_h_rec=25.0, tau_j=20.0, tau_u=0.5, refractory=40.0)
# L5 ET / CA1 pyramidal cell, two compartments, BAC firing (Larkum 1999).
L5 = dict(C=(150.0, 60.0), gL=(7.5, 3.0), g_sa=6.0, EL=-70.0, VT=-50.0, DT=2.0, a=2.0, b=80.0, tau_w=100.0,
          Vr=-58.0, t_ref=2.0, g_kv=10.0, bap=250.0, bac_amp=700.0, bac_theta=-35.0, bac_refractory=40.0,
          nmda_ratio=(0.5, 0.6))

SPIKE_V = -30.0
EXP_CLAMP = 10.0

# Background: Ornstein-Uhlenbeck current noise plus a tonic mean per population (the mean is fitted by G5).
OU_TAU = 5.0
OU_SD = {"E": 35.0, "I": 25.0}

# ------------------------------------------------------------------ populations
COLUMNS = ("S", "T", "P", "I")


@dataclass(frozen=True)
class Pop:
    name: str
    n: int
    kind: str          # "L23" (3-comp), "L5" (2-comp), or a POINT_KINDS key
    sign: str          # "E" | "I"
    region: str        # S | T | P | I | thal | hippo | nuclei
    mu: float = 0.0    # starting tonic background (pA), near rheobase; G5 calibration fits it
    cortical: bool = False   # has K_ATP (energy-gated)


def _column(c: str) -> list[Pop]:
    big = c in ("T", "P")
    n23 = 144 if big else 96
    n_pv23, n_sst23, n_vip = (18, 12, 12) if big else (12, 8, 8)
    n_et = 72 if c == "P" else 24
    n_it = 72 if c == "P" else (32 if c == "T" else 24)
    n_pv5, n_sst5 = (10, 10) if c == "P" else (6, 6)
    cx = dict(region=c, cortical=True)
    return [
        Pop(f"{c}.L23", n23, "L23", "E", mu=260.0, **cx),
        Pop(f"{c}.L5ET", n_et, "L5", "E", mu=180.0, **cx),
        Pop(f"{c}.NGF", 8, "NGF", "I", mu=100.0, **cx),
        Pop(f"{c}.PV23", n_pv23, "PV", "I", mu=150.0, **cx),
        Pop(f"{c}.SST23", n_sst23, "SST", "I", mu=50.0, **cx),
        Pop(f"{c}.VIP", n_vip, "VIP", "I", mu=45.0, **cx),
        Pop(f"{c}.L4", 40, "RS", "E", mu=130.0, **cx),
        Pop(f"{c}.PV4", 6, "PV", "I", mu=150.0, **cx),
        Pop(f"{c}.L5IT", n_it, "RS", "E", mu=130.0, **cx),
        Pop(f"{c}.PV5", n_pv5, "PV", "I", mu=150.0, **cx),
        Pop(f"{c}.SST5", n_sst5, "SST", "I", mu=50.0, **cx),
        Pop(f"{c}.L6CT", 32, "RS", "E", mu=125.0, **cx),
        Pop(f"{c}.PV6", 4, "PV", "I", mu=150.0, **cx),
        Pop(f"{c}.SST6", 4, "SST", "I", mu=50.0, **cx),
    ]


TC_CORE = {"S": 80, "T": 60, "I": 16, "P": 64}
TC_MTX = {"S": 14, "T": 16, "I": 14, "P": 72}


def populations() -> list[Pop]:
    pops: list[Pop] = []
    for c in COLUMNS:
        pops += _column(c)
    for c in COLUMNS:
        pops.append(Pop(f"{c}.TC", TC_CORE[c], "TC", "E", region="thal", mu=120.0))
        pops.append(Pop(f"{c}.MTX", TC_MTX[c], "TC", "E", region="thal", mu=120.0))
        pops.append(Pop(f"{c}.TRN", 10, "TRN", "I", region="thal", mu=130.0))
    pops.append(Pop("PVT", 8, "TC", "E", region="thal", mu=120.0))
    pops += [
        Pop("DG", 128, "DG", "E", region="hippo", mu=80.0),
        Pop("CA3", 96, "RS", "E", region="hippo", mu=130.0),
        Pop("CA3.PV", 12, "PV", "I", region="hippo", mu=150.0),
        Pop("CA1", 64, "L5", "E", region="hippo", mu=180.0),
        Pop("CA1.PV", 8, "PV", "I", region="hippo", mu=150.0),
        Pop("OLM", 8, "SST", "I", region="hippo", mu=50.0),
        Pop("VTA", 24, "DA", "E", region="nuclei"),
        Pop("RMTg", 8, "SST", "I", region="nuclei", mu=50.0),
        Pop("LC", 16, "PACE", "E", region="nuclei"),
        Pop("DRN", 16, "PACE", "E", region="nuclei"),
        Pop("NBM", 16, "PACE", "E", region="nuclei"),
        Pop("ORX", 12, "ORX", "E", region="nuclei", mu=110.0),
        Pop("PVN", 12, "RS", "E", region="nuclei", mu=125.0),
    ]
    # 3-compartment cells first, then 2-compartment, then point cells: dendritic targets are array prefixes.
    order = {"L23": 0, "L5": 1}
    return sorted(pops, key=lambda p: order.get(p.kind, 2))


# Named sub-ranges used by readouts and later milestones (relative to the population start).
ASSEMBLIES: dict[str, tuple[str, int, int]] = {
    **{f"T.motive{i}": ("T.L23", 16 * i, 16 * i + 16) for i in range(6)},
    **{f"T.claim{i}": ("T.L23", 96 + 6 * i, 102 + 6 * i) for i in range(8)},
    "T.liar": ("T.L5IT", 24, 32),
    **{f"P.fact{i}": ("P.L23", 12 * i, 12 * i + 12) for i in range(8)},
    **{f"P.agenda{i}": ("P.L23", 96 + 12 * i, 108 + 12 * i) for i in range(4)},
    "P.action": ("P.L5ET", 0, 24),
    "P.puppet.elara": ("P.L5ET", 24, 40),
    "P.puppet.marcus": ("P.L5ET", 40, 56),
    "P.bend": ("P.L5ET", 56, 72),
    **{f"P.counter{i}": ("P.L5IT", 24 + 4 * i, 28 + 4 * i) for i in range(12)},
    **{f"P.vavl{i}": ("P.MTX", 4 * i, 4 * i + 4) for i in range(18)},
    "CA1.ctx": ("CA1", 0, 32),            # the CA1 lines that reach the striatum
}

# ------------------------------------------------------------------ connectome


@dataclass(frozen=True)
class Projection:
    src: str                       # population, or "pop[a:b]"
    dst: str
    K: int                         # fixed in-degree
    recs: tuple[int, ...]
    comp: int                      # SOMA | BASAL | APICAL (point cells always receive at the soma)
    w: float                       # nS
    delay: tuple[int, int] = (1, 2)
    stp: int = 0                   # 0 none, 1 depressing, 2 facilitating
    plastic: str = ""              # "" | "gb" | "vogels"
    mod: str = ""                  # "" | "ach" (tuft AMPA scaled by 1 - 0.5 ACh)
    tags: tuple[str, ...] = field(default_factory=tuple)


def axon_delay(length_mm: float, velocity: float = 4.0) -> int:
    """Myelinated conduction delay, d = round(L / c + 0.5 ms), with c0 = 4 mm/ms."""
    return int(round(length_mm / velocity + 0.5))


DEP, FAC = 1, 2
A, AN = (AMPA,), (AMPA, NMDA)
FEEDFORWARD = (("S", "T"), ("S", "P"), ("T", "P"), ("I", "P"), ("S", "I"))


def projections() -> list[Projection]:
    P: list[Projection] = []

    def add(*args, **kw):
        P.append(Projection(*args, **kw))

    for c in COLUMNS:
        x = lambda p: f"{c}.{p}"  # noqa: E731
        gb_l4 = "gb" if c == "T" else ""
        gb_23 = "gb" if c in ("T", "P") else ""
        # thalamic core input (§7.2)
        add(x("TC"), x("L4"), 12, AN, SOMA, 1.2, (2, 4), DEP)
        add(x("TC"), x("PV4"), 8, A, SOMA, 1.6, (2, 4), DEP)
        add(x("TC"), x("L23"), 6, AN, BASAL, 0.9, (2, 4), DEP)
        add(x("TC"), x("L5ET"), 4, AN, SOMA, 0.8, (2, 4), DEP)
        # excitatory cortical
        add(x("L4"), x("L23"), 16, AN, BASAL, 1.0, (1, 2), DEP, gb_l4)
        add(x("L4"), x("L4"), 8, AN, SOMA, 0.5, (1, 2), DEP)
        add(x("L4"), x("PV23"), 8, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L4"), x("PV4"), 8, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L23"), x("L23"), 24, AN, BASAL, 0.7, (1, 3), DEP, gb_23)
        add(x("L23"), x("L5ET"), 16, AN, SOMA, 1.0, (1, 2), DEP)
        add(x("L23"), x("L5IT"), 16, AN, SOMA, 1.0, (1, 2), DEP)
        add(x("L23"), x("PV23"), 16, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L23"), x("SST23"), 16, A, SOMA, 0.9, (1, 2), FAC)
        add(x("L23"), x("VIP"), 8, A, SOMA, 0.6, (1, 2), FAC)
        add(x("L5IT"), x("L5ET"), 8, AN, SOMA, 0.8, (1, 2), DEP)
        add(x("L5ET"), x("L6CT"), 8, AN, SOMA, 0.8, (1, 2), DEP)
        add(x("L5IT"), x("L6CT"), 6, AN, SOMA, 0.6, (1, 2), DEP)
        add(x("L5ET"), x("PV5"), 8, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L5IT"), x("PV5"), 8, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L5ET"), x("SST5"), 8, A, SOMA, 0.9, (1, 2), FAC)
        add(x("L6CT"), x("PV6"), 8, A, SOMA, 1.6, (1, 2), DEP)
        add(x("L6CT"), x("SST6"), 6, A, SOMA, 0.9, (1, 2), FAC)
        # inhibition (Vogels iSTDP onto L2/3 PCs)
        add(x("PV23"), x("L23"), 10, (GABA_A,), SOMA, 3.0, plastic="vogels")
        add(x("PV4"), x("L4"), 8, (GABA_A,), SOMA, 3.0)
        add(x("PV23"), x("PV23"), 6, (GABA_A,), SOMA, 2.0)
        add(x("PV4"), x("PV4"), 4, (GABA_A,), SOMA, 2.0)
        add(x("SST23"), x("L23"), 8, (GABA_AS,), APICAL, 2.5, plastic="vogels")
        add(x("SST23"), x("PV23"), 4, (GABA_A,), SOMA, 1.5)
        add(x("VIP"), x("SST23"), 6, (GABA_A,), SOMA, 3.0)
        add(x("NGF"), x("L23"), 4, (GABA_AS, GABA_B), APICAL, 1.5)
        add(x("NGF"), x("L5ET"), 4, (GABA_AS, GABA_B), APICAL, 1.0)
        add(x("PV5"), x("L5ET"), 8, (GABA_A,), SOMA, 3.0)
        add(x("PV5"), x("L5IT"), 8, (GABA_A,), SOMA, 3.0)
        add(x("SST5"), x("L5ET"), 6, (GABA_AS,), APICAL, 2.5)
        add(x("PV6"), x("L6CT"), 6, (GABA_A,), SOMA, 3.0)
        add(x("SST6"), x("L6CT"), 4, (GABA_A,), SOMA, 2.0)
        # matrix thalamus into L1 (apical, ACh-gated)
        add(x("MTX"), x("L23"), 8, AN, APICAL, 1.2, (3, 5), mod="ach")
        add(x("MTX"), x("L5ET"), 8, AN, APICAL, 1.0, (3, 5), mod="ach")
        add(x("MTX"), x("NGF"), 6, A, SOMA, 1.5, (3, 5))
        add(x("MTX"), x("VIP"), 4, A, SOMA, 1.2, (3, 5))
        # corticothalamic loops
        add(x("L6CT"), x("TC"), 4, A, SOMA, 0.6, (3, 6))
        add(x("L6CT"), x("TRN"), 4, A, SOMA, 0.8, (3, 6))
        add(x("L5ET"), x("MTX"), 4, AN, SOMA, 1.0, (axon_delay(25), axon_delay(25)))
        add(x("TRN"), x("TC"), 8, (GABA_A, GABA_B), SOMA, 2.0)
        add(x("TRN"), x("MTX"), 6, (GABA_A,), SOMA, 2.0)
        add(x("TC"), x("TRN"), 6, A, SOMA, 1.0)
        add(x("TRN"), x("TRN"), 4, (GABA_A,), SOMA, 1.5)

    # inter-column (§7.3): feedforward L2/3 -> L4 and L2/3; feedback L5 IT -> apical tufts and NGF
    d40 = (axon_delay(40), axon_delay(40))
    for a, b in FEEDFORWARD:
        add(f"{a}.L23", f"{b}.L4", 4, AN, SOMA, 0.6, d40, DEP)
        add(f"{a}.L23", f"{b}.L23", 4, AN, BASAL, 0.5, d40, DEP)
        add(f"{b}.L5IT", f"{a}.L23", 6, AN, APICAL, 0.6, d40, DEP, mod="ach")
        add(f"{b}.L5IT", f"{a}.NGF", 4, A, SOMA, 0.6, d40, DEP)

    # hippocampus
    d60, d70 = (axon_delay(60),) * 2, (axon_delay(70),) * 2
    for c in ("S", "T"):
        add(f"{c}.L23", "DG", 2, AN, SOMA, 1.0, d60, DEP)
        add(f"{c}.L23", "CA3", 2, AN, SOMA, 0.6, d60, DEP)
    add("DG", "CA3", 2, AN, SOMA, 4.0, (2, 3), FAC)
    add("DG", "CA3.PV", 4, A, SOMA, 1.0, (1, 2), DEP)
    add("CA3", "CA3", 24, AN, SOMA, 0.6, (1, 3), DEP, "gb")
    add("CA3", "CA3.PV", 16, A, SOMA, 1.6, (1, 2), DEP)
    add("CA3.PV", "CA3", 10, (GABA_A,), SOMA, 3.0)
    add("CA3", "CA1", 20, AN, APICAL, 0.7, (2, 4), DEP, "gb")
    add("CA3", "CA1.PV", 8, A, SOMA, 1.0, (2, 4), DEP)
    add("CA1", "CA1.PV", 16, A, SOMA, 1.6, (1, 2), DEP)
    add("CA1.PV", "CA1", 10, (GABA_A,), SOMA, 3.0)
    add("CA1", "OLM", 16, A, SOMA, 0.9, (1, 2), FAC)
    add("OLM", "CA1", 6, (GABA_AS,), APICAL, 2.0)
    add("CA1", "P.L5IT", 8, AN, SOMA, 0.8, d70, DEP)
    add("CA1", "T.L23", 6, AN, BASAL, 0.6, d70, DEP)

    # neuromodulatory and hypothalamic nuclei
    d80, d90 = (axon_delay(80),) * 2, (axon_delay(90),) * 2
    add("P.L5IT", "VTA", 6, AN, SOMA, 0.8, d80)
    add("RMTg", "VTA", 6, (GABA_A,), SOMA, 2.0)
    add("P.L5ET", "LC", 4, AN, SOMA, 0.8, d90)
    add("P.L5ET", "DRN", 4, AN, SOMA, 0.8, d90)
    add("P.L5IT", "NBM", 4, AN, SOMA, 0.8, d90)
    add("I.L5ET", "ORX", 4, AN, SOMA, 0.8, d80)
    for dst in ("LC", "VTA", "PVT", "PVN"):
        add("ORX", dst, 4, AN, SOMA, 0.8, d40)
    add("PVT", "P.L5ET[24:72]", 6, AN, APICAL, 1.0, (axon_delay(30),) * 2)
    add("PVT", "P.NGF", 6, A, SOMA, 1.0, (axon_delay(30),) * 2)
    return P
