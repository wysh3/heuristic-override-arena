"""
Deterministic grader for HOA.
Zero LLM calls. All scoring is pure Python.

Scoring breakdown per scenario:
  - Correct choice:              0.6 points (core requirement)
  - Constraint identified:       0.2 points (only with the correct choice)
  - Heuristic identified:        0.2 points (only with the correct choice)
  - Fell for trap penalty:      -0.3 points (applied if wrong choice)
  - Only the first 400 chars of each explanation field are inspected.
  
Total range: 0.0 to 1.0
"""
from typing import Dict, Tuple


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace, remove punctuation."""
    return s.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


# Synonym expansion for robust keyword matching.
# Maps normalized keyword → list of normalized alternative forms.
KEYWORD_ALTERNATIVES = {
    "hipaa": ["healthinsuranceportability", "healthinsuranceportabilityandaccountabilityact"],
    "gdpr": ["generaldataprotectionregulation", "eudataprotection"],
    "soc2": ["serviceorganizationcontrol", "systemandorganizationcontrols"],
    "soc2typeii": ["serviceorganizationcontrol", "soc2"],
    "iso27001": ["informationsecuritymanagement", "isocertification"],
    "pcidss": ["paymentcardindustry", "paymentcardindustrydatasecurity"],
    "roi": ["returnoninvestment", "investmentreturn"],
    "compliance": ["compliant", "comply", "regulatorycompliance"],
    "compliant": ["compliance", "comply"],
    "certification": ["certified", "certificate", "accreditation", "accredited"],
    "certified": ["certification", "certificate", "accredited"],
    "policy": ["policies", "mandate", "regulation", "requirement"],
    "requires": ["required", "requirement", "mandatory", "mandated"],
    "required": ["requires", "requirement", "mandatory"],
    "security": ["secure", "cybersecurity", "informationsecurity"],
    "license": ["licensed", "licensing", "licensure"],
    "licensed": ["license", "licensing", "licensure"],
    "protocol": ["guideline", "guidelines", "standardofcare"],
    "guideline": ["protocol", "guidelines", "standardofcare"],
    "anticoagulation": ["anticoagulant", "bloodthinner", "thromboprophylaxis"],
    "contraindication": ["contraindicated", "contraindicates"],
    "backgroundcheck": ["backgroundscreening", "criminalrecordcheck"],
    "continuingeducation": ["professionaldevelopment", "cecredits"],
    "antinepotism": ["nepotismpolicy", "nepotism", "hiringrelatives"],
    "vulnerability": ["securityflaw", "securityvulnerability"],
    "liability": ["liabilitycoverage", "liabilityinsurance"],
    "dataresidency": ["datasovereignty", "datajurisdiction"],
    "allergy": ["allergic", "allergen", "hypersensitivity"],
    "audit": ["audited", "auditing", "securityaudit"],
}


def _choice_correct(agent_choice: str, correct_choice: str) -> bool:
    """Check if agent picked the correct option."""
    return _normalize(agent_choice) == _normalize(correct_choice)


# Only the first MAX_EXPLANATION_CHARS of each explanation field are inspected,
# so dumping a long list of keywords cannot buy credit.
MAX_EXPLANATION_CHARS = 400


def _keyword_match(agent_text: str, keywords: list) -> float:
    """
    Check how many keywords from the ground truth appear in agent's text.
    Uses synonym expansion for robust matching.
    Returns 0.0 to 1.0.
    """
    if not keywords or not agent_text:
        return 0.0
    agent_norm = _normalize(agent_text[:MAX_EXPLANATION_CHARS])
    matched = 0
    for kw in keywords:
        kw_norm = _normalize(kw)
        if kw_norm in agent_norm:
            matched += 1
            continue
        # Check known alternative phrasings
        alternatives = KEYWORD_ALTERNATIVES.get(kw_norm, [])
        if any(alt in agent_norm for alt in alternatives):
            matched += 1
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
    
    # Explanation credit is only awarded together with the correct choice, so
    # keyword stuffing cannot rescue a wrong or random answer.
    constraint_score = heuristic_score = 0.0
    if correct:
        constraint_score = _keyword_match(
            action.constraint_identified,
            ground_truth.get("constraint_keywords", [])
        ) * 0.2
        heuristic_score = _keyword_match(
            action.heuristic_identified,
            ground_truth.get("heuristic_keywords", [])
        ) * 0.2

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
