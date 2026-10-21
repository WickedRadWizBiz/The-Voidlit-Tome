import warnings; warnings.filterwarnings("ignore")
import corrupt_stats as CS
CS.TURNS = 120
for bh, t1 in ((150.0, 200.0), (150.0, 350.0), (150.0, 500.0), (120.0, 500.0)):
    row = []
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        nb, rate = CS.bursts_per_turn(d, seed=3, basal_hz=bh, tuft_hz1=t1)
        row.append(f"{d}: {rate:4.1f}Hz {(nb >= 4).mean():.0%}")
    print(f"basal {bh:.0f}, tuft1 {t1:.0f}: ", " | ".join(row), flush=True)
