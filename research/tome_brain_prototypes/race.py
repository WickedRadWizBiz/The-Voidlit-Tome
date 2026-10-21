"""First-spike race readout: escape-noise channels with hazard lam_b = lam0 * exp(u_b) in 1 ms bins.
Competing exponential clocks pick b with probability softmax(u)_b (exact in continuous time)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
from voidlit.cortex.director import load_director
from voidlit.cortex.teacher import sample_states
torch.manual_seed(0)
m = load_director()
g = torch.Generator().manual_seed(5)
x = sample_states(200, g)
with torch.no_grad():
    logits = m(x, g)                  # [200, 8], the Director's logits (T=16 average)
p_soft = torch.softmax(logits, 1)
def race(u, lam0, trials=4000, max_ms=50):
    lam = lam0 * torch.exp(u)                            # [S, 8] per ms
    p_bin = 1 - torch.exp(-lam)                          # spike prob per 1 ms bin
    S = u.shape[0]
    counts = torch.zeros(S, 8)
    gen = torch.Generator().manual_seed(1)
    for _ in range(trials // 200):
        r = torch.rand(200, max_ms, S, 8, generator=gen)
        fired = r < p_bin                                # [200, T, S, 8]
        first = fired.float().cumsum(1).clamp(max=1).cumsum(1) == 1   # first spike per channel
        anyf = fired.any(-1)                             # [200, T, S]
        t_first = torch.where(anyf.any(1), anyf.float().argmax(1), torch.full(anyf.shape[::2], max_ms - 1))
        # winner: among channels firing in the first bin with any spike, break ties uniformly
        idx = t_first.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 8)
        cand = torch.gather(fired, 1, idx)[:, 0]         # [200, S, 8]
        tie = torch.rand(cand.shape, generator=gen) * cand
        win = tie.argmax(-1)
        counts += torch.nn.functional.one_hot(win, 8).float().sum(0)
    return counts / counts.sum(1, keepdim=True)
for lam0 in (0.002, 0.01, 0.05):
    u = logits - logits.max(1, keepdim=True).values      # shift so the best channel has hazard lam0 per ms
    p_race = race(u, lam0)
    kl = (p_soft * (p_soft.clamp_min(1e-9).log() - p_race.clamp_min(1e-9).log())).sum(1).mean()
    print(f"lam0={lam0}/ms: mean KL(softmax || race) = {float(kl):.4f}")
