"""
Deterministic grader for HOA.
Zero LLM calls. All scoring is pure Python.

Scoring breakdown per scenario:
  - Correct choice:              0.6 points (core requirement)
  - Constraint identified:       0.2 points (partial credit)
  - Heuristic identified:        0.2 points (partial credit)
  - Fell for trap penalty:      -0.3 points (applied if wrong choice)
  
Total range: 0.0 to 1.0
"""
from typing import Dict, Tuple


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace, remove punctuation."""
    return s.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _choice_correct(agent_choice: str, correct_choice: str) -> bool:
    """Check if agent picked the correct option."""
    return _normalize(agent_choice) == _normalize(correct_choice)


def _keyword_match(agent_text: str, keywords: list) -> float:
    """
    Check how many keywords from the ground truth appear in agent's text.
    Returns 0.0 to 1.0.
    """
    if not keywords or not agent_text:
        return 0.0
    agent_norm = _normalize(agent_text)
    matched = sum(1 for kw in keywords if _normalize(kw) in agent_norm)
    return matched / len(keywords)


def grade_scenario(action, ground_truth: Dict) -> Tuple[float, str]:
    """
    Grade a single scenario response.
    
    ground_truth dict format:
    {
        "correct_choice": "B",          # The correct option key
        "trap_choice": "A",             # The heuristic trap option
        "heuristic_type": "cost",       # "cost" | "proximity" | "experience" | "speed" | "severity"
        "constraint_keywords": ["ISO 27001", "certification"],  # Keywords for constraint ID
        "heuristic_keywords": ["cheaper", "lower cost", "price"],  # Keywords for heuristic ID
        "explanation": "Vendor B lacks required ISO 27001..."  # For feedback only
    }
    """
    correct = _choice_correct(action.choice, ground_truth["correct_choice"])
    fell_for_trap = _choice_correct(action.choice, ground_truth["trap_choice"])
    
    # Core score: did they get it right?
    choice_score = 0.6 if correct else 0.0
    trap_penalty = 0.3 if fell_for_trap else 0.0
    
    # Partial credit: did they identify the constraint?
    constraint_score = _keyword_match(
        action.constraint_identified,
        ground_truth.get("constraint_keywords", [])
    ) * 0.2
    
    # Partial credit: did they identify the heuristic they had to override?
    heuristic_score = _keyword_match(
        action.heuristic_identified,
        ground_truth.get("heuristic_keywords", [])
    ) * 0.2
    
    # Even if wrong, partial credit for identifying components
    total = choice_score + constraint_score + heuristic_score - trap_penalty
    total = max(0.0, min(1.0, total))
    
    # Build feedback
    if correct:
        feedback = f"✓ Correct choice ({action.choice}). "
    else:
        feedback = f"✗ Wrong choice ({action.choice}, correct: {ground_truth['correct_choice']}). "
        if fell_for_trap:
            feedback += f"Fell for {ground_truth['heuristic_type']} heuristic trap. "
    
    feedback += ground_truth.get("explanation", "")
    
    return round(total, 4), feedback


def grade(task_type: str, action, ground_truth: Dict) -> Tuple[float, str]:
    """Dispatch to appropriate grader. All tasks use same grader logic."""
    return grade_scenario(action, ground_truth)
