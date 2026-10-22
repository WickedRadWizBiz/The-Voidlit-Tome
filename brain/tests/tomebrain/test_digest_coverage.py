"""The digest covers everything a save holds (M6 step 2).

``Cosim.digest`` is the oracle of every bit-exactness gate (G6b-d, G7, the save tests): two brains with one digest must
have one future. So moving any value a save holds by one ulp (any array of any partition, the basal ganglia's named
scalars, the habit cortex, the noise streams' seeds, the bus and the clock) must change it.
Mutant: a basal-ganglia digest that leaves out kappa_h fails.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.cortex.teacher import sample_states  # noqa: E402
from voidlit.tomebrain import persist  # noqa: E402
from voidlit.tomebrain.brain import TomeBrain  # noqa: E402
from voidlit.tomebrain.partition import state_arrays  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402


@pytest.fixture(scope="module")
def brain():
    """A brain two turns in (a gift, then a punished turn), so its learning traces and tonic levels have moved."""
    b = TomeBrain(3)
    for i, x in enumerate(sample_states(2, torch.Generator().manual_seed(5)).numpy()):
        b.mind.reward = -0.09 * i
        b.set_us([{"gift": 1.0}, {"harm": 1.0}][i])
        b.set_input(director_features=x, senses=np.full(8, 0.3, np.float32),
                    body=np.array([0.7, 0.9, 0.2], np.float32), facts=np.linspace(0.2, 0.9, 8, dtype=np.float32))
        b.phase_a()
        b.phase_b(x)
    return b


def _leaves(d: dict, path: tuple = ()):
    """The path of every number or array of numbers in a nested save."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, path + (k,))
        elif np.asarray(v).dtype.kind in "biuf":
            yield path + (k,)


def _nudged(v):
    """``v`` with its last element one ulp up (a whole number one up, a flag flipped)."""
    a = np.array(v, copy=True)
    flat = a.reshape(-1)
    if a.dtype.kind == "f":
        flat[-1] = np.nextafter(flat[-1], np.inf, dtype=a.dtype)
    elif a.dtype.kind == "b":
        flat[-1] = not flat[-1]
    else:
        flat[-1] += 1
    return a if isinstance(v, np.ndarray) else a.item()


def unseen(brain) -> list[tuple]:
    """Every value of the brain's save that the digest does not see: moved by one ulp, the digest stays the same."""
    base = persist.snapshot(brain.cosim)
    want = brain.digest()
    paths = [("partitions", *p) for p in _leaves(base["partitions"])]
    paths += [("bus", "buf", k) for k in base["bus"]["buf"]] + [("clock", "t_ms"), ("clock", "turn")]
    out = []
    try:
        for path in paths:
            *parents, last = path
            d = base
            for k in parents:
                d = d[k]
            held = d[last]
            d[last] = _nudged(held)
            persist.apply(brain.cosim, base)
            if brain.digest() == want:
                out.append(path)
            d[last] = held
    finally:
        persist.apply(brain.cosim, base)
    assert brain.digest() == want
    return out


def test_the_digest_sees_every_value_a_save_holds(brain):
    assert unseen(brain) == []


def test_mutant_a_digest_without_kappa_h_fails(brain, monkeypatch):
    def without_kappa_h(self):
        d = self.state_dict()
        del d["scalars"]["kappa_h"]
        return state_arrays(d)

    monkeypatch.setattr(BG.SJPartition, "digest_arrays", without_kappa_h)
    assert unseen(brain) == [("partitions", "sj", "scalars", "kappa_h")]
