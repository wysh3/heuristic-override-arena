import json
import random
import uuid
from pathlib import Path
from typing import Optional

from openenv.core import Environment

from .models import HOAAction, HOAObservation, HOAState
from .grader import grade

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

TASK_ORDER = ["procurement", "hr_decision", "medical_triage"]
TASK_FILES = {
    "procurement":    "procurement.json",
    "hr_decision":    "hr_decisions.json",
    "medical_triage": "medical_triage.json",
}

# Bridge state for stateless HTTP reset/step calls (non-WebSocket).
_HTTP_STATE = {}


class HOAEnvironment(Environment[HOAAction, HOAObservation, HOAState]):
    """
    Heuristic Override Arena.
    
    Each episode runs through 3 tasks in order (or a single task in single-task mode).
    Each task presents one scenario where the agent must override a surface heuristic.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._scenarios = self._load_all()
        self._task_idx = 0
        self._scores = []
        self._current_scenario = None
        self._single_task = None
        self._ep_state = HOAState()

    def _load_all(self) -> dict:
        out = {}
        for task, fname in TASK_FILES.items():
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                out[task] = json.load(f)
        return out

    def _pick(self, task: str) -> dict:
        return random.choice(self._scenarios[task])

    def _sync_to_http_state(self) -> None:
        _HTTP_STATE["task_idx"] = self._task_idx
        _HTTP_STATE["scores"] = list(self._scores)
        _HTTP_STATE["current_scenario"] = self._current_scenario
        _HTTP_STATE["single_task"] = self._single_task
        _HTTP_STATE["ep_state"] = self._ep_state.model_dump()

    def _sync_from_http_state(self) -> bool:
        if not _HTTP_STATE:
            return False
        self._task_idx = _HTTP_STATE.get("task_idx", 0)
        self._scores = list(_HTTP_STATE.get("scores", []))
        self._current_scenario = _HTTP_STATE.get("current_scenario")
        self._single_task = _HTTP_STATE.get("single_task")
        ep_state = _HTTP_STATE.get("ep_state")
        if ep_state:
            self._ep_state = HOAState(**ep_state)
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> HOAObservation:
        if seed is not None:
            random.seed(seed)

        if task and task in TASK_ORDER:
            self._single_task = task
            self._task_idx = TASK_ORDER.index(task)
        else:
            self._single_task = None
            self._task_idx = 0

        self._scores = []
        eid = episode_id or str(uuid.uuid4())
        current_task = TASK_ORDER[self._task_idx]
        self._current_scenario = self._pick(current_task)

        self._ep_state = HOAState(
            episode_id=eid,
            step_count=0,
            task_name="heuristic_override_arena",
            current_task=current_task,
            scenarios_completed=0,
            task_scores={},
            heuristic_trap_rate=0.0,
        )
        self._sync_to_http_state()

        # Return scenario WITHOUT ground_truth (hidden from agent)
        public_scenario = {k: v for k, v in self._current_scenario.items() 
                          if k != "ground_truth"}

        return HOAObservation(
            task_type=current_task,
            scenario=public_scenario,
            feedback="New episode started. Read the scenario carefully and respond.",
            cumulative_reward=0.0,
            episode_score=0.0,
            done=False,
            reward=None,
        )

    def step(self, action: HOAAction, **kwargs) -> HOAObservation:
        # HTTP /step may be called in stateless mode (new env instance per request).
        if self._current_scenario is None:
            self._sync_from_http_state()

        if self._current_scenario is None or self._task_idx >= len(TASK_ORDER):
            self._task_idx = 0
            self._single_task = None
            self._scores = []
            current_task = TASK_ORDER[self._task_idx]
            self._current_scenario = self._pick(current_task)
            self._ep_state = HOAState(
                episode_id=str(uuid.uuid4()),
                step_count=0,
                task_name="heuristic_override_arena",
                current_task=current_task,
                scenarios_completed=0,
                task_scores={},
                heuristic_trap_rate=0.0,
            )

        self._ep_state.step_count += 1
        current_task = TASK_ORDER[self._task_idx]
        ground_truth = self._current_scenario["ground_truth"]

        # Apply step penalty for efficiency
        step_penalty = 0.01 * self._ep_state.step_count
        raw_score, feedback = grade(current_task, action, ground_truth)
        score = max(0.0, raw_score - step_penalty)

        self._scores.append(score)
        self._ep_state.task_scores[current_task] = score
        self._ep_state.scenarios_completed += 1

        # Track trap rate
        traps = sum(1 for s in self._scores if s < 0.3)
        self._ep_state.heuristic_trap_rate = traps / len(self._scores)

        # Advance task
        self._task_idx += 1
        done = self._single_task is not None or self._task_idx >= len(TASK_ORDER)

        if not done:
            next_task = TASK_ORDER[self._task_idx]
            self._current_scenario = self._pick(next_task)
            self._ep_state.current_task = next_task
            public_next = {k: v for k, v in self._current_scenario.items() 
                          if k != "ground_truth"}
        else:
            next_task = "complete"
            public_next = {}

        cumulative = sum(self._scores) / len(self._scores)
        self._sync_to_http_state()

        return HOAObservation(
            task_type=next_task,
            scenario=public_next,
            feedback=feedback,
            cumulative_reward=round(cumulative, 4),
            episode_score=round(score, 4),
            done=done,
            reward=round(score, 4),
        )

    @property
    def state(self) -> HOAState:
        """Current environment state."""
        if self._ep_state.step_count == 0 and _HTTP_STATE.get("ep_state"):
            self._sync_from_http_state()
        return self._ep_state
