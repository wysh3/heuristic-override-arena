import json
import random
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from openenv.core import Environment

from .models import HOAAction, HOAObservation, HOAState
from .grader import grade

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

# Core tasks (original 3 domains)
TASK_ORDER = ["procurement", "hr_decision", "medical_triage"]
TASK_FILES = {
    "procurement":    "procurement.json",
    "hr_decision":    "hr_decisions.json",
    "medical_triage": "medical_triage.json",
}

# Extended tasks (cognitive biases)
EXTENDED_TASKS = ["cognitive_biases", "edge_cases"]
EXTENDED_FILES = {
    "cognitive_biases": "cognitive_biases.json",
    "edge_cases": "edge_cases.json",
}

# Difficulty levels with trap intensities
DIFFICULTY_CONFIG = {
    "easy": {"trap_weight": 0.5, "tasks": ["procurement"]},
    "medium": {"trap_weight": 0.75, "tasks": ["hr_decision", "cognitive_biases"]},
    "hard": {"trap_weight": 1.0, "tasks": ["medical_triage", "edge_cases"]},
}

# Number of scenarios per episode in single-task mode
SCENARIOS_PER_EPISODE = 3

# Heuristic type categories for analytics
HEURISTIC_CATEGORIES = {
    "economic": ["cost", "speed", "efficiency"],
    "social": ["authority_bias", "affinity_bias", "social_pressure", "familiarity_bias", "similarity_bias"],
    "cognitive": ["sunk_cost", "recency_bias", "status_quo_bias", "availability_bias", "anecdotal_bias"],
    "clinical": ["severity", "urgency", "proximity"],
    "professional": ["experience", "performance", "prestige_bias", "popularity_bias"],
}

# Bridge state for stateless HTTP reset/step calls (non-WebSocket).
_HTTP_STATE = {}


class HOAEnvironment(Environment[HOAAction, HOAObservation, HOAState]):
    """
    Heuristic Override Arena.
    
    An OpenEnv environment that trains AI agents to override surface heuristics
    when explicit constraints require it. Based on CMU research (arXiv:2603.29025).
    
    Features:
    - 100 scenarios across 5 domains (75 original + 25 cognitive bias/edge cases)
    - 15+ heuristic trap types (cost, authority, sunk cost, recency, etc.)
    - Curriculum learning support (easy → medium → hard progression)
    - Per-bias performance analytics
    - Multi-step episodes for stronger RL signal
    
    Episode Modes:
    - Single task: reset(task="procurement") - 1 scenario from that domain
    - Full episode: reset() - 3 scenarios (easy → medium → hard progression)
    - Curriculum: reset(difficulty="easy") - scenarios filtered by difficulty
    - Extended: reset(task="cognitive_biases") - cognitive bias scenarios
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._scenarios = self._load_all()
        self._task_idx = 0
        self._scores = []
        self._current_scenario = None
        self._single_task = None
        self._difficulty = None
        self._ep_state = HOAState()
        self._bias_stats: Dict[str, Dict[str, float]] = {}  # Per-bias performance
        self._used_scenario_ids: set = set()  # Dedup within episode

    def _load_all(self) -> dict:
        """Load all scenarios from JSON files."""
        out = {}
        # Load core tasks
        for task, fname in TASK_FILES.items():
            path = SCENARIOS_DIR / fname
            if path.exists():
                with open(path) as f:
                    out[task] = json.load(f)
        # Load extended tasks
        for task, fname in EXTENDED_FILES.items():
            path = SCENARIOS_DIR / fname
            if path.exists():
                with open(path) as f:
                    out[task] = json.load(f)
        return out

    def _pick(self, task: str, difficulty: Optional[str] = None, exclude_ids: Optional[set] = None) -> dict:
        """
        Pick a random scenario from the specified task.
        Optionally filter by difficulty and exclude already-used scenarios.
        """
        scenarios = self._scenarios.get(task, [])
        if not scenarios:
            # Fallback to any task that has scenarios
            for t in self._scenarios:
                if self._scenarios[t]:
                    scenarios = self._scenarios[t]
                    break
        
        if not scenarios:
            # Fatal: no scenarios found anywhere
            return {
                "id": "fallback_000",
                "context": "Scenario system failure. Please contact environment developer.",
                "question": "Choose A.",
                "options": {"A": "A", "B": "B"},
                "ground_truth": {"correct_choice": "A", "trap_choice": "B", "heuristic_type": "none"}
            }

        valid_pool = scenarios
        if difficulty:
            filtered = [s for s in valid_pool 
                       if s.get("trap_intensity", "medium") == difficulty]
            if filtered:
                valid_pool = filtered
        
        if exclude_ids:
            filtered = [s for s in valid_pool if s.get("id") not in exclude_ids]
            if filtered:
                valid_pool = filtered
        
        return random.choice(valid_pool)
    
    def _get_task_for_difficulty(self, difficulty: str) -> str:
        """Get a task appropriate for the difficulty level."""
        config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
        available_tasks = [t for t in config["tasks"] if t in self._scenarios and self._scenarios[t]]
        if not available_tasks:
            # Fallback to absolute list of all tasks
            available_tasks = [t for t in self._scenarios if self._scenarios[t]]
        return random.choice(available_tasks) if available_tasks else "procurement"
    
    def _categorize_heuristic(self, heuristic_type: str) -> str:
        """Categorize a heuristic type for analytics."""
        for category, types in HEURISTIC_CATEGORIES.items():
            if heuristic_type in types:
                return category
        return "other"
    
    def _update_bias_stats(self, heuristic_type: str, score: float) -> None:
        """Track per-bias performance for analytics."""
        if heuristic_type not in self._bias_stats:
            self._bias_stats[heuristic_type] = {"total": 0, "sum": 0.0}
        self._bias_stats[heuristic_type]["total"] += 1
        self._bias_stats[heuristic_type]["sum"] += score
    
    def get_bias_performance(self) -> Dict[str, float]:
        """Return average score per heuristic type."""
        return {
            bias: stats["sum"] / stats["total"] 
            for bias, stats in self._bias_stats.items()
            if stats["total"] > 0
        }
    
    def get_available_tasks(self) -> List[str]:
        """Return list of all available task types."""
        return list(self._scenarios.keys())

    def _sync_to_http_state(self) -> None:
        _HTTP_STATE["task_idx"] = self._task_idx
        _HTTP_STATE["scores"] = list(self._scores)
        _HTTP_STATE["current_scenario"] = self._current_scenario
        _HTTP_STATE["single_task"] = self._single_task
        _HTTP_STATE["difficulty"] = self._difficulty
        _HTTP_STATE["ep_state"] = self._ep_state.model_dump()
        _HTTP_STATE["bias_stats"] = self._bias_stats

    def _sync_from_http_state(self) -> bool:
        if not _HTTP_STATE:
            return False
        self._task_idx = _HTTP_STATE.get("task_idx", 0)
        self._scores = list(_HTTP_STATE.get("scores", []))
        self._current_scenario = _HTTP_STATE.get("current_scenario")
        self._single_task = _HTTP_STATE.get("single_task")
        self._difficulty = _HTTP_STATE.get("difficulty")
        self._bias_stats = _HTTP_STATE.get("bias_stats", {})
        ep_state = _HTTP_STATE.get("ep_state")
        if ep_state:
            self._ep_state = HOAState(**ep_state)
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs,
    ) -> HOAObservation:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode ID
            task: Specific task to run ("procurement", "hr_decision", etc.)
                  If None, runs full 3-task episode with curriculum
            difficulty: Filter scenarios by difficulty ("easy", "medium", "hard")
                       Used with task=None for curriculum learning
        
        Returns:
            Initial observation with first scenario
        """
        if seed is not None:
            random.seed(seed)

        self._difficulty = difficulty
        all_tasks = TASK_ORDER + [t for t in EXTENDED_TASKS if t in self._scenarios]

        if task and task in all_tasks:
            # Single-task mode
            self._single_task = task
            self._task_idx = all_tasks.index(task) if task in all_tasks else 0
            current_task = task
        elif difficulty:
            # Difficulty-based curriculum mode
            self._single_task = None
            self._task_idx = 0
            current_task = self._get_task_for_difficulty(difficulty)
        else:
            # Full multi-task episode
            self._single_task = None
            self._task_idx = 0
            current_task = TASK_ORDER[0]

        self._scores = []
        self._bias_stats = {}
        self._used_scenario_ids = set()
        eid = episode_id or str(uuid.uuid4())
        self._current_scenario = self._pick(current_task, difficulty)
        self._used_scenario_ids.add(self._current_scenario.get("id"))

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
        """
        Execute one step in the environment.
        
        Process the agent's action, calculate rewards, and advance to next scenario
        or end the episode.
        """
        # HTTP /step may be called in stateless mode (new env instance per request).
        if self._current_scenario is None:
            self._sync_from_http_state()

        all_tasks = TASK_ORDER + [t for t in EXTENDED_TASKS if t in self._scenarios]

        if self._current_scenario is None or self._task_idx >= len(all_tasks):
            self._task_idx = 0
            self._single_task = None
            self._scores = []
            current_task = TASK_ORDER[self._task_idx]
            self._current_scenario = self._pick(current_task, self._difficulty)
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
        current_task = self._single_task or (
            TASK_ORDER[self._task_idx] if self._task_idx < len(TASK_ORDER) 
            else all_tasks[self._task_idx]
        )
        ground_truth = self._current_scenario["ground_truth"]
        
        # Track per-bias performance
        heuristic_type = ground_truth.get("heuristic_type", "unknown")

        # Apply step penalty for efficiency
        step_penalty = 0.01 * self._ep_state.step_count
        raw_score, feedback = grade(current_task, action, ground_truth)
        score = max(0.0, raw_score - step_penalty)
        
        # Update bias statistics
        self._update_bias_stats(heuristic_type, score)

        self._scores.append(score)
        self._ep_state.task_scores[f"{current_task}_step{self._ep_state.step_count}"] = score
        self._ep_state.scenarios_completed += 1

        # Track trap rate
        traps = sum(1 for s in self._scores if s < 0.3)
        self._ep_state.heuristic_trap_rate = traps / len(self._scores)

        # Determine if episode is done
        if self._single_task is not None:
            # Single-task mode: run SCENARIOS_PER_EPISODE scenarios
            done = self._ep_state.scenarios_completed >= SCENARIOS_PER_EPISODE
        else:
            self._task_idx += 1
            done = self._task_idx >= len(TASK_ORDER)

        if not done:
            # Get next scenario
            if self._single_task:
                next_task = self._single_task
            elif self._difficulty:
                next_task = self._get_task_for_difficulty(self._difficulty)
            else:
                next_task = TASK_ORDER[self._task_idx]
            self._current_scenario = self._pick(next_task, self._difficulty, self._used_scenario_ids)
            self._used_scenario_ids.add(self._current_scenario.get("id"))
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
