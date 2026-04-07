"""Integration tests for HOAEnvironment reset/step flow."""
import pytest

from hoa_env.environment import HOAEnvironment, SCENARIOS_PER_EPISODE
from hoa_env.models import HOAAction


def make_action(choice="A", constraint="", heuristic=""):
    return HOAAction(
        choice=choice,
        constraint_identified=constraint,
        heuristic_identified=heuristic,
    )


class TestEnvironmentReset:
    def test_reset_returns_scenario(self):
        env = HOAEnvironment()
        obs = env.reset(seed=42, task="procurement")
        assert obs.task_type == "procurement"
        assert obs.scenario
        assert "context" in obs.scenario
        assert "options" in obs.scenario
        assert obs.done is False

    def test_ground_truth_hidden(self):
        env = HOAEnvironment()
        obs = env.reset(seed=42, task="procurement")
        assert "ground_truth" not in obs.scenario

    def test_all_tasks_loadable(self):
        env = HOAEnvironment()
        tasks = env.get_available_tasks()
        assert "procurement" in tasks
        assert "hr_decision" in tasks
        assert "medical_triage" in tasks
        assert "cognitive_biases" in tasks
        assert "edge_cases" in tasks

    def test_difficulty_filter(self):
        env = HOAEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.task_type == "procurement"
        assert obs.done is False


class TestEnvironmentStep:
    def test_step_returns_reward_in_range(self):
        env = HOAEnvironment()
        env.reset(seed=42, task="procurement")
        obs = env.step(make_action(choice="A"))
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_single_task_multi_step(self):
        """Single-task mode should run SCENARIOS_PER_EPISODE steps."""
        env = HOAEnvironment()
        obs = env.reset(seed=42, task="procurement")

        for i in range(SCENARIOS_PER_EPISODE):
            assert obs.done is False, f"Episode ended early at step {i}"
            obs = env.step(make_action(choice="A"))

        assert obs.done is True

    def test_multi_step_different_scenarios(self):
        """Each step should present a different scenario."""
        env = HOAEnvironment()
        obs = env.reset(seed=42, task="procurement")
        seen_contexts = [obs.scenario.get("context", "")]

        for _ in range(SCENARIOS_PER_EPISODE - 1):
            obs = env.step(make_action(choice="A"))
            if obs.scenario:
                seen_contexts.append(obs.scenario.get("context", ""))

        # At least 2 unique contexts (dedup should prevent repeats)
        unique = set(seen_contexts)
        assert len(unique) >= 2, f"Got only {len(unique)} unique scenarios"

    def test_multi_task_episode(self):
        """No-task reset should run 3-task progression."""
        env = HOAEnvironment()
        obs = env.reset(seed=42)
        assert obs.task_type == "procurement"

        obs = env.step(make_action(choice="A"))
        assert obs.done is False
        assert obs.task_type == "hr_decision"

        obs = env.step(make_action(choice="B"))
        assert obs.done is False
        assert obs.task_type == "medical_triage"

        obs = env.step(make_action(choice="A"))
        assert obs.done is True

    def test_cumulative_reward_computed(self):
        env = HOAEnvironment()
        env.reset(seed=42, task="procurement")
        obs = env.step(make_action(choice="A"))
        assert obs.cumulative_reward is not None
        assert obs.cumulative_reward >= 0.0


class TestEnvironmentScoring:
    def test_correct_choice_scores_higher(self):
        """Correct choice must score higher than trap choice."""
        env = HOAEnvironment()
        env.reset(seed=42, task="procurement")
        gt = env._current_scenario["ground_truth"]

        # Score with correct choice
        env2 = HOAEnvironment()
        env2.reset(seed=42, task="procurement")
        obs_correct = env2.step(make_action(choice=gt["correct_choice"]))

        # Score with trap choice
        env3 = HOAEnvironment()
        env3.reset(seed=42, task="procurement")
        obs_trap = env3.step(make_action(choice=gt["trap_choice"]))

        assert obs_correct.reward > obs_trap.reward
