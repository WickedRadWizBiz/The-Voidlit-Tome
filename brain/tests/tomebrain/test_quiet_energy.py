"""The quiet interval solves the window's whole ATP equation (design §5.6, §6.7), synaptic cost included:

    da/dt = k_sup g CBF (1.05 - a) - k_rest - k_syn g_exc - k_spike r

Each cell's excitatory conductance starts at its end-of-turn value and relaxes with the activity profile, so a cell
still bombarded at the end of a turn pays for it, and one whose inputs fall silent stops paying.
"""

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402

G_EXC = 5.0      # nS of AMPA + NMDA on the target cell: about what a fed P/I cell carries at the end of a turn


@pytest.fixture(scope="module")
def tiny():
    pops = [PR.Pop("A", 2, "RS", "E", region="nuclei"), PR.Pop("B", 2, "RS", "E", region="nuclei")]
    net = NW.build(pops=pops, projs=[PR.Projection("A", "B", 1, PR.AN, PR.SOMA, 1.0)])
    return net, NW.make_quiet(net)


def quiet_atp(tiny, rate_turn: float, rate_rest: float):
    net, quiet = tiny
    st = NW.init_state(net, jax.random.PRNGKey(0))
    b = net.idx("B").start
    st["G"] = st["G"].at[net.slot(b, PR.SOMA, PR.AMPA)].set(1.0).at[net.slot(b, PR.SOMA, PR.NMDA)].set(G_EXC - 1.0)
    out = quiet(st, jnp.asarray(1.0, jnp.float32), jnp.full(net.n, rate_turn, jnp.float32),
                jnp.full(net.n, rate_rest, jnp.float32))
    return [float(x) for x in out["atp"][b:b + 2]]


def test_quiet_charges_the_synaptic_cost_of_a_turn_that_ends_busy(tiny):
    E = NW.ENERGY
    loaded, unloaded = quiet_atp(tiny, 0.0, 0.0)            # no activity change: the conductance stays as it ended
    assert unloaded == pytest.approx(1.05 - E["k_rest"] / E["k_sup"], abs=2e-3)
    assert loaded == pytest.approx(1.05 - (E["k_rest"] + E["k_syn"] * G_EXC) / E["k_sup"], abs=2e-3)


def test_quiet_conductance_relaxes_with_the_activity(tiny):
    """The turn ends busy (10 Hz) and the network falls silent at rest: the inputs, and their cost, fade with tau_post
    long before the 20 s are over."""
    loaded, unloaded = quiet_atp(tiny, 10.0, 0.0)
    assert loaded == pytest.approx(unloaded, abs=2e-3)
