# SpikingJelly partition timing: cerebellar forward model (MF -> GrC/Golgi -> PF -> PC/MLI -> DCN -> IO), dt = 0.5 ms,
# multi-step mode, 5-ms sync windows (10 steps per window), 50 windows per turn.
import time, torch, warnings
warnings.filterwarnings("ignore")
from spikingjelly.activation_based import neuron, functional, surrogate
torch.manual_seed(0); torch.set_num_threads(1)
T = 10
n_mf, n_grc, n_go, n_pc, n_mli, n_dcn, n_io = 64, 600, 20, 24, 24, 12, 12
# granule cells: 4 mossy-fibre "claws" each (realistic convergence)
W_mf_grc = torch.zeros(n_mf, n_grc)
for j in range(n_grc): W_mf_grc[torch.randperm(n_mf)[:4], j] = 0.6
W_go_grc = -0.4 * (torch.rand(n_go, n_grc) < 0.3).float()
W_grc_go = 0.05 * (torch.rand(n_grc, n_go) < 0.2).float()
W_pf_pc = 0.02 * torch.rand(n_grc, n_pc)                              # plastic (Marr-Albus-Ito)
W_pf_mli = 0.03 * (torch.rand(n_grc, n_mli) < 0.3).float()
W_mli_pc = -0.5 * torch.eye(n_mli)
W_pc_dcn = -0.8 * torch.kron(torch.eye(n_dcn), torch.ones(2, 1))
W_dcn_io = -0.6 * torch.eye(n_dcn)
mk = lambda tau, vth=1.0: neuron.LIFNode(tau=tau, v_threshold=vth, surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="s")
grc, go, pc, mli, dcn, io = mk(4.0), mk(10.0), mk(5.0), mk(5.0), mk(8.0), mk(20.0)
state = {k: torch.zeros(1, n) for k, n in (("grc", n_grc), ("go", n_go), ("pc", n_pc), ("mli", n_mli), ("dcn", n_dcn), ("io", n_io))}
elig = torch.zeros(n_grc, n_pc)
@torch.no_grad()
def window(mf, actual):
    global elig
    out = []
    for t in range(T):
        s = state
        s_grc = grc(mf[t] @ W_mf_grc + s["go"] @ W_go_grc)
        s_go = go(mf[t] @ (0.1*torch.ones(n_mf, n_go)) + s_grc @ W_grc_go)
        s_mli = mli(s_grc @ W_pf_mli)
        s_pc = pc(0.35 + s_grc @ W_pf_pc + s_mli @ W_mli_pc)          # tonic intrinsic drive -> high simple-spike rate
        s_dcn = dcn(0.3 + s_pc @ W_pc_dcn + 0.05 * mf[t, :, :n_dcn])
        s_io = io(actual[t] + s_dcn @ W_dcn_io)
        # PF eligibility (~100 ms window) and CF-gated LTD / PF-alone LTP
        elig.mul_(0.995).add_(s_grc.T.expand(-1, n_pc) * 1.0)
        cf = s_io.repeat_interleave(2, dim=1)
        W_pf_pc.add_(-1e-3 * elig * cf + 1e-5 * s_grc.T * (1 - cf)).clamp_(0, 0.1)
        state.update(grc=s_grc, go=s_go, pc=s_pc, mli=s_mli, dcn=s_dcn, io=s_io)
        out.append(s_dcn)
    return torch.stack(out)
mf = (torch.rand(T, 1, n_mf) < 0.05).float(); act = (torch.rand(T, 1, n_io) < 0.01).float()
window(mf, act)
for rep in range(3):
    t0 = time.perf_counter()
    for k in range(50): window(mf, act)
    if rep == 2: print(f"SpikingJelly cerebellum epoch (50 x 10 steps @0.5 ms): {1e3*(time.perf_counter()-t0):.0f} ms")
