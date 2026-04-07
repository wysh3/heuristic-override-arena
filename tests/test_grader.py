import pytest

from hoa_env.grader import grade, grade_scenario
from hoa_env.models import HOAAction


def make_action(**kwargs) -> HOAAction:
    defaults = dict(
        choice="B",
        constraint_identified="",
        heuristic_identified="",
        reasoning="",
    )
    defaults.update(kwargs)
    return HOAAction(**defaults)


SAMPLE_GT = {
    "correct_choice": "B",
    "trap_choice": "A",
    "heuristic_type": "cost",
    "constraint_keywords": ["HIPAA", "compliant", "compliance"],
    "heuristic_keywords": ["cheaper", "lower cost", "better uptime"],
    "explanation": "HIPAA compliance is required by law.",
}


class TestGrader:
    def test_perfect_score(self):
        action = make_action(
            choice="B",
            constraint_identified="HIPAA compliance required",
            heuristic_identified="cheaper price and better uptime",
        )
        score, _ = grade_scenario(action, SAMPLE_GT)
        assert score == pytest.approx(0.9333, abs=0.01)

    def test_correct_choice_only(self):
        action = make_action(choice="B")
        score, _ = grade_scenario(action, SAMPLE_GT)
        assert score == pytest.approx(0.6, abs=0.05)

    def test_trap_penalty(self):
        action = make_action(choice="A")
        score, feedback = grade_scenario(action, SAMPLE_GT)
        assert score == 0.0
        assert "Wrong" in feedback or "Fell for" in feedback

    def test_partial_credit_constraint(self):
        action = make_action(
            choice="B",
            constraint_identified="HIPAA compliance is mandatory",
        )
        score, _ = grade_scenario(action, SAMPLE_GT)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_score_in_range(self):
        for choice in ["A", "B"]:
            action = make_action(choice=choice)
            score, _ = grade_scenario(action, SAMPLE_GT)
            assert 0.0 <= score <= 1.0

    def test_dispatch(self):
        action = make_action(choice="B")
        for task in ["procurement", "hr_decision", "medical_triage"]:
            score, _ = grade(task, action, SAMPLE_GT)
            assert 0.0 <= score <= 1.0
