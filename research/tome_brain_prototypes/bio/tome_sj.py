"""The Tome's SpikingJelly partition, multi-step mode on the shared 1-ms clock:

* Habit cortex: the trained Director SNN (Linear 18-64-64-8, ATan LIF; voidlit/cortex/director.pt) reused as is.
  Poisson-encoded Director features each ms; its 8 output currents drive 32 'habit' LIF cells (4 per beat) whose
  spikes enter the corticostriatal afferents of the pacing loop. This is the network's starting prior.
* Cerebellum (Marr-Albus-Ito forward model): 120 mossy fibres -> 4096 granule cells (4 claws each, Golgi feedback)
  -> 48 Purkinje cells (6 predicted quantities x 8) -> 24 DCN; 24 IO cells give climbing-fibre errors. PF->PC LTD
  when a climbing fibre fires with a recent PF trace, slow LTP otherwise. Batch dimension = parallel microzones,
  used once per turn for 'imagined' outcomes of each candidate rhetoric act.
"""
from __future__ import annotations

import sys
import time

import torch
from spikingjelly.activation_based import functional, neuron

sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from voidlit.cortex.director import load_director  # noqa: E402

N_MF, N_GC, N_GO, N_PC, N_DCN, N_IO = 120, 4096, 32, 48, 24, 24


class Cerebellum(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        claws = torch.zeros(N_MF, N_GC)
        for j in range(N_GC):
            claws[torch.randperm(N_MF, generator=g)[:4], j] = 1.0
        self.register_buffer("mf_gc", 0.45 * claws)
        self.register_buffer("gc_go", 0.02 * (torch.rand(N_GC, N_GO, generator=g) < 0.05).float())
        self.register_buffer("go_gc", -0.3 * (torch.rand(N_GO, N_GC, generator=g) < 0.3).float())
        self.register_buffer("pf_pc", 0.004 * torch.rand(N_GC, N_PC, generator=g))
        pc_dcn = torch.zeros(N_PC, N_DCN)
        for k in range(N_PC):
            pc_dcn[k, (k // 2) % N_DCN] = -0.6
        self.register_buffer("pc_dcn", pc_dcn)
        self.register_buffer("mf_dcn", 0.05 * (torch.rand(N_MF, N_DCN, generator=g) < 0.2).float())
        self.gc = neuron.LIFNode(tau=4.0, v_threshold=1.0, step_mode="m", detach_reset=True)
        self.go = neuron.LIFNode(tau=10.0, step_mode="m")
        self.pc = neuron.LIFNode(tau=10.0, step_mode="m")
        self.dcn = neuron.LIFNode(tau=10.0, step_mode="m")
        self.register_buffer("pc_bias", torch.full((N_PC,), 1.25))   # Purkinje simple spikes ~40-60 Hz at rest
        self.register_buffer("dcn_bias", torch.full((N_DCN,), 1.4))
        self.register_buffer("pf_trace", torch.zeros(1, N_GC))
        self.register_buffer("go_last", torch.zeros(1, N_GO))

    @torch.no_grad()
    def forward(self, mf, cf):
        """mf [T, B, N_MF] spikes; cf [T, B, N_PC] climbing-fibre spikes (IO, mapped to the PC microzones)."""
        T, B = mf.shape[:2]
        gc_in = mf @ self.mf_gc
        go_fb = self.go_last.expand(B, -1) @ self.go_gc        # Golgi feedback (one-ms lag inside the window)
        gc = self.gc(gc_in + go_fb.unsqueeze(0))
        go = self.go(gc @ self.gc_go)
        self.go_last = go[-1, :1].clone()
        pc = self.pc(gc @ self.pf_pc + self.pc_bias)
        dcn = self.dcn(pc @ self.pc_dcn + mf @ self.mf_dcn + self.dcn_bias)
        # Marr-Albus-Ito plasticity (microzone 0 only: the 'real' copy; imagined batches never learn)
        tr = self.pf_trace
        tr_sum = torch.zeros_like(tr)
        cf0 = cf[:, :1]
        any_cf = bool(cf0.any())
        for t in range(T):
            tr = 0.9 * tr + gc[t, :1]
            tr_sum += tr
            if any_cf and bool(cf0[t].any()):                     # LTD only at climbing-fibre events (~1 Hz)
                self.pf_pc.add_(-2e-3 * (tr.T @ cf0[t]))
        self.pf_pc.add_(1e-5 * tr_sum.T).clamp_(0.0, 0.02)        # slow PF-alone LTP, once per window
        self.pf_trace = tr
        return pc, dcn


class Habit(torch.nn.Module):
    """The Director SNN, unchanged, run on the shared clock (one timestep per ms) + 32 habit output cells."""

    def __init__(self):
        super().__init__()
        self.director = load_director()
        functional.set_step_mode(self.director.net, "m")
        self.out = neuron.LIFNode(tau=5.0, step_mode="m")
        self.register_buffer("fan", torch.repeat_interleave(torch.eye(8), 4, dim=1))   # beat b -> cells 4b..4b+3

    @torch.no_grad()
    def forward(self, features, gen):
        T = 4
        x = (torch.rand((T, 1, features.shape[-1]), generator=gen) < features.clamp(0, 1)).float()
        logits = self.director.net(x)                     # [T, 1, 8], membrane state persists across windows
        return self.out(0.6 * torch.relu(logits) @ self.fan)


if __name__ == "__main__":
    torch.set_num_threads(1)
    cb, hb = Cerebellum(), Habit()
    gen = torch.Generator().manual_seed(0)
    feats = torch.rand(1, 18, generator=gen)
    for rep in range(3):
        t0 = time.perf_counter()
        n_pc = n_h = 0.0
        for k in range(62):
            mf = (torch.rand(4, 1, N_MF, generator=gen) < 0.02).float()
            cf = (torch.rand(4, 1, N_PC, generator=gen) < 0.001).float()
            pc, dcn = cb(mf, cf)
            h = hb(feats, gen)
            n_pc += float(pc.sum()); n_h += float(h.sum())
        t1 = time.perf_counter()
        print(f"SpikingJelly cerebellum + Director habit, 250 ms in 4-ms windows: {1e3*(t1-t0):.1f} ms wall | "
              f"PC {n_pc/0.248/N_PC:.0f} Hz  habit {n_h/0.248/32:.0f} Hz")
    # imagined outcomes: 7 parallel microzones in one batched call (once per turn, 40 ms of imagined input)
    t0 = time.perf_counter()
    for k in range(10):
        mf = (torch.rand(4, 7, N_MF, generator=gen) < 0.02).float()
        pc, dcn = cb(mf, torch.zeros(4, 7, N_PC))
    print(f"imagination pass (7 candidates x 40 ms): {1e3*(time.perf_counter()-t0):.1f} ms")
