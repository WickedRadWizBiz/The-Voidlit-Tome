"""Gate G3a: the astrocyte is a calcium-excitable cell that frequency-codes its neurons' activity (De Pitta 2009)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402


def astro_events(spikes_per_ms: float, seconds: int = 120, gamma_jump: float | None = None):
    """Drive one domain with a steady spike rate; count Ca2+ events (upward crossings of 0.5 uM)."""
    a = jnp.asarray([[0.1], [0.05], [0.8], [0.0], [0.0]], jnp.float32)
    step = jax.jit(lambda a, d: NW.astro_step(a, d))
    rng = np.random.default_rng(0)
    events, gamma_max, prev = 0, 0.0, 0.05
    drive = rng.poisson(spikes_per_ms, seconds * 1000).astype(np.float32)
    for t in range(seconds * 1000):
        a = step(a, jnp.asarray([drive[t]]))
        if t % 10 == 0:
            c = float(a[1, 0])
            events += prev < NW.ASTRO["gamma_theta"] <= c
            prev = c
            gamma_max = max(gamma_max, float(a[3, 0]))
    return events, gamma_max


@pytest.fixture(scope="module")
def sweep():
    # quiet, a domain of 12 L2/3 PCs at about 2 Hz, and the same domain driven hard
    return {rate: astro_events(rate) for rate in (0.0, 0.02, 0.08)}


def test_g3a_no_events_without_glutamate(sweep):
    assert sweep[0.0] == (0, 0.0)


def test_g3a_sustained_glutamate_gives_events_and_gliotransmitter(sweep):
    events, gamma = sweep[0.02]
    assert events >= 2 and gamma > 0.3


def test_g3a_event_frequency_codes_the_drive(sweep):
    assert sweep[0.08][0] > sweep[0.02][0]
