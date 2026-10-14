import warnings

import pytest

warnings.filterwarnings("ignore")

from voidlit.cortex.director import Director, load_director  # noqa: E402
from voidlit.limbic import BranchSeed, NPCSpec, PlotSpec, TacticSpec  # noqa: E402
from voidlit.tome.content import Codex, load_place  # noqa: E402


@pytest.fixture(scope="session")
def codex():
    return Codex.load()


@pytest.fixture(scope="session")
def lantern():
    return load_place("hollow_lantern")


@pytest.fixture(scope="session")
def director_model():
    model = load_director()
    assert model is not None, "voidlit/cortex/director.pt is missing; run `python -m voidlit train-director`"
    return model


@pytest.fixture
def director(director_model):
    return Director(director_model, seed=0)


def make_spec(npc_id: str = "elara", numeric_id: int = 1) -> NPCSpec:
    return NPCSpec(
        id=npc_id, name=npc_id.title(), numeric_id=numeric_id,
        plots=[PlotSpec("shelter", "Shelter the player", "ally", 0.6),
               PlotSpec("codes", "Extract their codes", "extract", 0.35),
               PlotSpec("sell", "Sell them out", "betray", 0.15)],
        tactics=[TacticSpec("warm", "Be warm.", "shelter", ["warm line"]),
                 TacticSpec("probe", "Probe.", "codes", ["probe line"]),
                 TacticSpec("cold", "Deflect coldly.", "sell", ["cold line"])],
        branches=[BranchSeed("shelter", ["rapport_pos", "leverage"], 0.6),
                  BranchSeed("codes", ["leverage", "exposure"], 0.55),
                  BranchSeed("sell", ["aggression", "deception", "rapport_neg"], 0.5)],
        trust={},
    )


@pytest.fixture
def spec():
    return make_spec()
