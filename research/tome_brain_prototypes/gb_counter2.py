from gb_counter import turn, gap
for f in (20.0, 40.0):
    for k in (4, 6, 8, 12, 16):
        rho = 0.05
        for _ in range(k):
            rho = gap(turn(rho, True, f=f))
        a = rho
        traj = []
        for i in range(300):
            rho = gap(turn(rho, False))
            if i in (9, 49, 299): traj.append(round(rho, 3))
        print(f"f={f:.0f} Hz, {k:2d} abusive turns: rho {a:.3f} -> after 10/50/300 quiet turns {traj}")
