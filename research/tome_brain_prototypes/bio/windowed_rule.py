"""WindowedMSTDPET: BindsNET's MSTDPET (Florian 2007, eqs. 2.7-2.8, lagged eligibility) evaluated once per sync
window instead of once per millisecond. Spikes are recorded each step (two vector copies); at the end of the window
the traces, eligibility and eligibility trace are advanced in closed form with four small matmuls. When the reward
is constant within a window (it is: dopamine arrives through the bus once per window) the result equals running
MSTDPET step by step, up to the per-step [wmin, wmax] clamp (applied once per window here).

Optional per-postsynaptic dopamine sign (+1 D1/VS, -1 D2) implements the inverted indirect-pathway rule.
"""
from __future__ import annotations

import torch
from bindsnet.learning import LearningRule


class WindowedMSTDPET(LearningRule):
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0, window: int = 4,
                 post_sign: torch.Tensor | None = None, **kwargs):
        super().__init__(connection=connection, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)
        self.W = int(window)
        n_pre, n_post = self.source.n, self.target.n
        self.tc_plus = float(kwargs.get("tc_plus", 20.0))
        self.tc_minus = float(kwargs.get("tc_minus", 20.0))
        self.tc_z = float(kwargs.get("tc_e_trace", 25.0))
        self.a_plus = float(kwargs.get("a_plus", 1.0))
        self.a_minus = float(kwargs.get("a_minus", -1.0))
        self.post_sign = post_sign
        self.buf_pre = torch.zeros(self.W, n_pre)
        self.buf_post = torch.zeros(self.W, n_post)
        self.k = 0
        self.p_plus = torch.zeros(n_pre)
        self.p_minus = torch.zeros(n_post)
        self.e_fac = (torch.zeros(n_pre), torch.zeros(n_post), torch.zeros(n_pre), torch.zeros(n_post))
        self.z = torch.zeros(n_pre, n_post)
        self.reward_acc = 0.0
        self.update = self._connection_update   # BindsNET dispatches per connection type this way

    def _connection_update(self, **kwargs) -> None:
        if not kwargs.get("learning", True):
            return
        self.buf_pre[self.k] = self.source.s.view(-1).float()
        self.buf_post[self.k] = self.target.s.view(-1).float()
        self.reward_acc = float(kwargs.get("reward", 0.0))
        self.k += 1
        if self.k == self.W:
            self.flush(self.reward_acc)

    @torch.no_grad()
    def flush(self, reward: float) -> None:
        W, k = self.W, self.k
        dp, dm = torch.exp(torch.tensor(-1.0 / self.tc_plus)), torch.exp(torch.tensor(-1.0 / self.tc_minus))
        beta = torch.exp(torch.tensor(-1.0 / self.tc_z))
        pre, post = self.buf_pre[:k], self.buf_post[:k]
        P, Q = torch.empty_like(pre), torch.empty_like(post)
        pp, pm = self.p_plus, self.p_minus
        for t in range(k):
            pp = dp * pp + self.a_plus * pre[t]
            pm = dm * pm + self.a_minus * post[t]
            P[t], Q[t] = pp, pm
        # eligibility e_{j-1}, j = 1..k: e_0 is carried from the last window (factored), then steps 1..k-1
        j = torch.arange(1, k + 1, dtype=torch.float32)
        c_sum = (1 - beta ** (k - j + 1)) / (1 - beta)          # weight of e_{j-1} in sum_k z_k
        c_end = beta ** (k - j)                                  # weight of e_{j-1} in z_k (end of window)
        a, b, c, d = self.e_fac
        e0 = torch.outer(a, b) + torch.outer(c, d)

        def mix(cw):
            m = cw[0] * e0
            if k > 1:
                cc = cw[1:]
                m = m + (P[:-1] * cc[:, None]).T @ post[:-1] + (pre[:-1] * cc[:, None]).T @ Q[:-1]
            return m / self.tc_z
        s0 = (beta * (1 - beta ** k) / (1 - beta))
        z_sum = self.z * s0 + mix(c_sum)
        self.z = self.z * beta ** k + mix(c_end)
        dw = self.nu[0] * self.connection.dt * reward * z_sum
        if self.post_sign is not None:
            dw = dw * self.post_sign
        w = self.connection.w
        w += dw
        w.clamp_(self.connection.wmin, self.connection.wmax)
        self.p_plus, self.p_minus = pp, pm
        self.e_fac = (P[-1].clone(), post[-1].clone(), pre[-1].clone(), Q[-1].clone())
        self.k = 0
