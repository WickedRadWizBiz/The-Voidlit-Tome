"""Gates G6 (one clock, one bus, exact) and G7 (save and load continue bit-exactly).

Two torch partitions stand in for the SpikingJelly and BindsNET parts here: SpikingJelly LIF populations driven
by the cortex's L5 IT cells over the bus, inhibiting the P column's matrix thalamus back over the bus (a GPi-like
loop), with counter-based noise.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from voidlit.tomebrain.bus import CausalityError, SpikeBus  # noqa: E402
from voidlit.tomebrain.clock import Cosim  # noqa: E402
from voidlit.tomebrain.partition import BusEdge, JaxPartition, Partition  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402
from voidlit.tomebrain.persist import load, save  # noqa: E402
from voidlit.tomebrain.rng import Streams  # noqa: E402

L5IT = ["S.L5IT", "T.L5IT", "P.L5IT", "I.L5IT"]
N_L5IT = 24 + 32 + 72 + 24


class LIFProbe(Partition):
    BIAS = 0.75     # sized for the metabolic cortex's L5 IT (about 3 Hz): sparse probe spikes, never silent
    def __init__(self, name: str, run_seed: int, n: int = 64, delay: int = 5):
        from spikingjelly.activation_based import neuron

        torch.set_num_threads(1)
        self.name, self.n, self.delay = name, n, delay
        self.outputs = {f"{name}.out": n}
        self.lif = neuron.LIFNode(tau=10.0, v_threshold=1.0, v_reset=0.0, detach_reset=True)
        g = torch.Generator().manual_seed(7 if name == "sj" else 11)    # anatomy: fixed, not the run seed
        self.w = torch.rand(N_L5IT, n, generator=g) * 0.6
        self.streams = Streams(run_seed, "sj" if name == "sj" else "bn")
        self.lif.v = torch.zeros(n)

    def step(self, bus, t0, W):
        x = torch.from_numpy(bus.read("J.L5IT", t0, W, self.delay))
        out = []
        for i in range(W):
            noise = torch.from_numpy(self.streams.normal("v", t0 + i, (self.n,)))
            out.append(self.lif(x[i] @ self.w + self.BIAS + 0.25 * noise))
        return {f"{self.name}.out": torch.stack(out).numpy()}

    def state_dict(self):
        return {"v": self.lif.v.numpy().copy(), "streams": self.streams.state_dict()}

    def load_state_dict(self, d):
        self.lif.v = torch.from_numpy(np.array(d["v"], np.float32))
        self.streams = Streams.from_state(d["streams"])

    def digest_arrays(self):
        return [self.lif.v.numpy()]


def brain(run_seed=0, W=4, order=("sj", "bn")):
    edges = [BusEdge(f"{p}.out", 64, "P.MTX", 8, (PR.GABA_A,), PR.SOMA, 2.5, 5) for p in ("sj", "bn")]
    jp = JaxPartition(run_seed, edges=edges, outputs={"J.L5IT": L5IT, "J.PMTX": "P.MTX"})
    cos = Cosim(jp, [LIFProbe("sj", run_seed), LIFProbe("bn", run_seed)], W=W)
    cos.order = list(order)
    return cos


RECORD = ("J.L5IT", "J.PMTX", "sj.out", "bn.out")


# ------------------------------------------------------------------ G6a
def test_g6a_bus_refuses_rows_from_the_future_and_delivers_on_time():
    bus = SpikeBus({"a": 3}, W=4)
    with pytest.raises(CausalityError):
        bus.read("a", 0, 4, 2)                  # shorter than a window
    row = np.zeros((4, 3), np.float32)
    row[1, 2] = 1.0                             # a spike at tick 1
    bus.write("a", 0, row)
    with pytest.raises(CausalityError):
        bus.read("a", 8, 4, 4)                  # needs ticks 4..7, which nobody has written yet
    got = bus.read("a", 4, 4, 6)                # ticks 4..7 through a 6-ms axon read ticks -2..1
    assert got[3, 2] == 1.0 and got.sum() == 1.0
    bus.write("a", 4, np.zeros((4, 3), np.float32))
    assert bus.read("a", 8, 4, 4).sum() == 0
    with pytest.raises(CausalityError):
        bus.write("a", 16, row)                 # ticks must be contiguous


# ------------------------------------------------------------------ G6b
@pytest.mark.slow
def test_g6b_windowed_equals_lockstep_bitwise():
    a, b = brain(W=4), brain(W=1)
    ra, rb = a.run(480, RECORD), b.run(480, RECORD)
    for k in RECORD:
        assert ra[k].sum() > 0, k
        assert np.array_equal(ra[k], rb[k]), k


# ------------------------------------------------------------------ G6c, G6d
@pytest.mark.slow
def test_g6c_async_equals_sequential_and_g6d_order_does_not_matter():
    a, b, c = brain(), brain(), brain(order=("bn", "sj"))
    for _ in range(3):
        a.run(120)
        b.run(120, async_jax=False)
        c.run(120)
    assert a.digest() == b.digest() == c.digest()


# ------------------------------------------------------------------ G6e
def test_g6e_seeds():
    a, b, c = brain(run_seed=3), brain(run_seed=3), brain(run_seed=4)
    for x in (a, b, c):
        x.run(120)
    assert a.digest() == b.digest() != c.digest()


# ------------------------------------------------------------------ G7
@pytest.mark.slow
def test_g7_save_mid_run_and_continue_in_a_different_seed_instance(tmp_path):
    a = brain(run_seed=5)
    a.run(200)
    save(a, tmp_path / "tome_brain.pt")
    b = brain(run_seed=99)                  # built with another run seed: load must restore everything
    load(b, tmp_path / "tome_brain.pt")
    assert a.digest() == b.digest()
    for _ in range(2):                      # two more turns' worth of brain time
        ra, rb = a.run(240, RECORD), b.run(240, RECORD)
        for k in RECORD:
            assert np.array_equal(ra[k], rb[k]), k
    assert a.digest() == b.digest()


def test_g7_load_refuses_a_different_connectome(tmp_path):
    from voidlit.tomebrain.jaxpart import network as NW

    a = brain()
    save(a, tmp_path / "b.pt")
    other = Cosim(JaxPartition(0, outputs={"J.L5IT": L5IT}, net=NW.build(seed=1)), [], W=4)
    with pytest.raises(ValueError):
        load(other, tmp_path / "b.pt")
