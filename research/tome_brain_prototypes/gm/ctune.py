import warnings; warnings.filterwarnings("ignore")
import corrupt_stats as CS
CS.TURNS = 120
for bh in (150.0, 300.0, 450.0):
    for t1 in (1500.0, 3000.0):
        row = []
        for d in (0.0, 0.5, 1.0):
            nb, rate = CS.bursts_per_turn(d, seed=3, basal_hz=bh, tuft_hz1=t1)
            row.append(f"d={d}: {rate:5.1f}Hz att {(nb >= 4).mean():.0%}")
        print(f"basal {bh:.0f} Hz, tuft1 {t1:.0f}: ", " | ".join(row), flush=True)
