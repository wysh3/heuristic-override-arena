from typing import Any, Dict

from pydantic import Field
from openenv.core import Action, Observation, State


class HOAAction(Action):
    """
    Agent's decision in a heuristic override scenario.
    
    The agent must:
    1. Identify the correct choice (overriding the obvious heuristic)
    2. Identify what the heuristic trap was
    3. Identify the actual constraint that determines the correct answer
    """
    choice: str = ""
    # The agent's selected option. Must match one of the option keys in the scenario
    # e.g. "A", "B", "vendor_alpha", "candidate_1" etc.
    # Case-insensitive matching will be used in grader
    
    constraint_identified: str = ""
    # The agent's identification of the key constraint that overrides the heuristic
    # e.g. "ISO 27001 certification required", "security clearance mandatory"
    # Used for partial credit scoring
    
    heuristic_identified: str = ""
    # The agent's identification of the misleading heuristic
    # e.g. "cheaper price", "more experience", "faster delivery"
    # Used for partial credit scoring
    
    reasoning: str = ""
    # Chain of thought — not graded but useful for debugging and analysis


class HOAObservation(Observation):
    """
    Observation presented to the agent.
    Inherits done: bool and reward: Optional[float] from base class.
    """
    task_type: str = ""
    # "procurement" | "hr_decision" | "medical_triage" | "complete"
    
    scenario: Dict[str, Any] = Field(default_factory=dict)
    # Full scenario dict from JSON file
    # Contains: context, options, question, (hidden: ground_truth)
    
    feedback: str = ""
    # Human-readable result of last action (not used for grading)
    
    cumulative_reward: float = 0.0
    
    episode_score: float = 0.0
    # Running average of rewards this episode


class HOAState(State):
    """
    Episode state.
    Inherits episode_id: Optional[str] and step_count: int from base class.
    """
    task_name: str = "heuristic_override_arena"
    current_task: str = ""
    scenarios_completed: int = 0
    task_scores: Dict[str, float] = Field(default_factory=dict)
    heuristic_trap_rate: float = 0.0
    # Fraction of scenarios where agent fell for the trap (lower = better)
