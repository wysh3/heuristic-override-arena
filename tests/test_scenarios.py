"""Data-integrity checks for the scenario files."""
import json
from pathlib import Path

import pytest

SCEN = Path(__file__).parent.parent / "scenarios"
FILES = sorted(SCEN.glob("*.json"))


def _all():
    for f in FILES:
        for s in json.loads(f.read_text()):
            yield f.name, s


def test_total_is_100():
    assert sum(1 for _ in _all()) == 100


def test_ids_unique():
    ids = [s["id"] for _, s in _all()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("fname,s", list(_all()), ids=lambda v: v if isinstance(v, str) else v["id"])
def test_scenario_well_formed(fname, s):
    gt = s["ground_truth"]
    assert gt["correct_choice"] in s["options"]
    assert gt["trap_choice"] in s["options"]
    assert gt["correct_choice"] != gt["trap_choice"]
    assert gt["constraint_keywords"] and gt["heuristic_keywords"]
    assert gt.get("explanation")
    assert s["context"] and s["question"]
