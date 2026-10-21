import time, torch, warnings
warnings.filterwarnings("ignore")
from spikingjelly.activation_based import neuron, functional, surrogate
torch.manual_seed(0); torch.set_num_threads(1)
T = 10
n_mf, n_grc, n_go, n_pc, n_mli, n_dcn, n_io = 64, 600, 20, 24, 24, 12, 12
W_mf_grc = torch.zeros(n_mf, n_grc)
for j in range(n_grc): W_mf_grc[torch.randperm(n_mf)[:4], j] = 0.6
W_go_grc = -0.4 * (torch.rand(n_go, n_grc) < 0.3).float(); W_grc_go = 0.05 * (torch.rand(n_grc, n_go) < 0.2).float()
W_pf_pc = 0.02 * torch.rand(n_grc, n_pc); W_pf_mli = 0.03 * (torch.rand(n_grc, n_mli) < 0.3).float()
W_mli_pc = -0.5 * torch.eye(n_mli); W_pc_dcn = -0.8 * torch.kron(torch.eye(n_dcn), torch.ones(2, 1)); W_dcn_io = -0.6 * torch.eye(n_dcn)
mk = lambda tau: neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="m")
pops = dict(grc=mk(4.0), go=mk(10.0), pc=mk(5.0), mli=mk(5.0), dcn=mk(8.0), io=mk(20.0))
net = torch.nn.ModuleDict(pops).eval()
print("backend:", pops["grc"].backend, "| training:", pops["grc"].training)
prev = dict(go=torch.zeros(T, 1, n_go), io=torch.zeros(T, 1, n_io), pc=torch.zeros(T,1,n_pc))
elig = torch.zeros(n_grc, n_pc); decay = torch.tensor(0.995) ** torch.arange(T - 1, -1, -1).float()
@torch.no_grad()
def window(mf, actual):
    global elig
    golgi_inh = prev["go"].mean(0, keepdim=True) @ W_go_grc                    # Golgi acts with >= 5 ms delay (slow GABA)
    s_grc = pops["grc"](mf @ W_mf_grc + golgi_inh)                             # [T, 1, n_grc]
    s_go = pops["go"](mf @ (0.1 * torch.ones(n_mf, n_go)) + s_grc @ W_grc_go)
    s_mli = pops["mli"](s_grc @ W_pf_mli)
    s_pc = pops["pc"](0.35 + s_grc @ W_pf_pc + s_mli @ W_mli_pc)
    s_dcn = pops["dcn"](0.3 + s_pc @ W_pc_dcn + 0.05 * mf[:, :, :n_dcn])
    s_io = pops["io"](actual + prev["io"].mean(0, keepdim=True) * 0 + s_dcn.mean(0, keepdim=True) @ W_dcn_io)
    # Marr-Albus-Ito: PF eligibility (window-summed, exponentially weighted), CF-gated LTD, PF-alone LTP
    pf = (s_grc[:, 0, :] * decay[:, None]).sum(0)
    elig.mul_(0.995 ** T).add_(pf[:, None])
    cf = s_io[:, 0, :].sum(0).repeat_interleave(2)
    W_pf_pc.add_(-1e-3 * elig * cf[None, :] + 1e-5 * pf[:, None] * (cf[None, :] == 0)).clamp_(0, 0.1)
    prev.update(go=s_go, io=s_io, pc=s_pc)
    return s_dcn
mf = (torch.rand(T, 1, n_mf) < 0.05).float(); act = (torch.rand(T, 1, n_io) < 0.01).float()
window(mf, act)
for rep in range(3):
    t0 = time.perf_counter()
    for k in range(50): window(mf, act)
    if rep == 2: print(f"SpikingJelly cerebellum, multi-step eval: epoch = {1e3*(time.perf_counter()-t0):.0f} ms")
