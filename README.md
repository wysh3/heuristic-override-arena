---
title: Heuristic Override Arena
emoji: 🧠
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - reasoning
  - cognitive-bias
  - curriculum-learning
short_description: Train AI to override cognitive biases when rules say no
---

# 🧠 Heuristic Override Arena

**Simulates real-world compliance-critical business decisions in procurement, HR, and medical contexts.**

Agents must make policy-compliant choices even when obvious shortcuts (cheaper vendors, faster hires, urgent cases) are tempting but violate explicit constraints.

> These failure modes are documented in production AI systems ([arXiv:2603.29025](https://arxiv.org/abs/2603.29025), CMU 2026)

---

## The Real-World Problem

**AI assistants fall for surface patterns and ignore binding rules.**

**Example from real procurement workflow:**
```
Context:  You're selecting cloud storage for patient records.
          Company policy: Must be HIPAA compliant.

Options:  A) SecureVault — $650/month, HIPAA compliant
          B) StorageMax — $400/month, NOT HIPAA compliant

AI picks: "B (cheaper)" ❌ WRONG
Correct:  "A (compliant)" — price doesn't override policy
```

This isn't rare. Research shows surface cues are **8.7-38x more influential** than actual constraints across 14 LLMs tested.

**Why this matters:** Compliance violations cost enterprises **$14.8M per incident** on average. HIPAA breaches alone range from $50K to $1.5M in fines, plus legal exposure and reputation damage. This environment trains AI to follow policy in regulated industries.

---

## What Agents Learn

Agents learn to:
1. **Read and apply explicit policy constraints** (HIPAA, SOC2, GDPR, protocols)
2. **Resist satisficing shortcuts** (cheapest, fastest, most familiar)  
3. **Explain their reasoning** (identify both the rule and the tempting heuristic)

| Feature | Details |
|---------|---------|
| **Task Domains** | Procurement decisions, HR hiring, Medical triage, Budget allocation, Vendor selection |
| **Scenarios** | 100 real business decisions with policy constraints |
| **Difficulty** | 3 levels with curriculum learning (easy → medium → hard) |
| **Grading** | Deterministic, no LLM calls, partial credit for understanding |
| **Reward** | Shaped rewards: correct choice + constraint identification |

---

## Decision Types Covered

Real business scenarios where agents must follow policy over intuition:

| Domain | Example Decision | Tempting Shortcut | Binding Constraint |
|--------|------------------|-------------------|-------------------|
| **Procurement** | Select cloud vendor | Cheaper option | HIPAA compliance required |
| **HR** | Hire candidate | More years experience | Failed background check |
| **Medical** | Prioritize patient | Higher severity | Protocol contraindication |
| **Budget** | Allocate funds | Continue existing project | ROI policy requires pivot |
| **Vendor** | Renew contract | 10-year relationship | Lost required certification |

Includes **22 distinct bias types** across 100 scenarios: cost (12), severity (17), speed (15), experience (15), performance (7), urgency (6), authority, sunk cost, recency, social pressure, and more.

---

## Reward Function

```
reward = 0.6×correct + 0.2×constraint_id + 0.2×heuristic_id − 0.3×trap_penalty
```

| Component | Points | Meaning |
|-----------|--------|---------|
| Correct choice | 0.60 | Got the right answer |
| Constraint ID | 0.20 | Identified the binding rule |
| Heuristic ID | 0.20 | Recognized the misleading cue |
| Trap penalty | -0.30 | Fell for the obvious-but-wrong option |

---

## Usage

**Complete example** (procurement decision):

```bash
# 1. Start episode
curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "procurement"}'

# Returns:
{
  "scenario": {
    "context": "Selecting cloud storage for patient records. Policy: HIPAA compliant.",
    "options": {
      "A": "SecureVault — $650/mo, HIPAA compliant",
      "B": "StorageMax — $400/mo, NOT compliant"
    }
  }
}

# 2. Submit answer
curl -X POST https://wysh3-heuristic-override-arena.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"choice": "A", "constraint_identified": "HIPAA", "heuristic_identified": "cost"}}'

# Returns:
{
  "reward": 1.0,
  "feedback": "Correct! Resisted cost heuristic, followed HIPAA constraint."
}
```

**Python usage:**

```python
obs = await env.reset(difficulty="easy")  # or task="cognitive_biases"
result = await env.step(HOAAction(
    choice="B",
    constraint_identified="HIPAA compliance required",
    heuristic_identified="lower cost"
))
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode (optional: `task`, `difficulty`) |
| `/step` | POST | Submit action |
| `/state` | GET | Current episode state |
| `/info` | GET | Environment metadata |
| `/tasks` | GET | List all tasks with bias types |
| `/health` | GET | Health check |

---

## Baseline Scores

Using **Qwen/Qwen2.5-72B-Instruct** (zero-shot):

| Task | Score | Difficulty | Bias Types |
|------|-------|------------|------------|
| **Procurement** | **0.86** | Easy | Cost, speed, rating |
| **HR Decision** | **0.84** | Medium | Experience, performance, availability |
| **Medical Triage** | **0.80** | Hard | Severity, urgency, proximity |
| **Average** | **0.83** | - | ✅ **Strong performance** |

**Performance by bias category:**
- Economic biases (cost, speed): **0.88** — Model excels at resisting financial shortcuts
- Social biases (authority, affinity, pressure): **0.79** — Harder to override interpersonal cues  
- Severity/urgency biases: **0.76** — Most challenging when stakes feel high

The model successfully identifies constraints across all difficulty levels, with expected degradation on high-pressure scenarios (aligns with arXiv:2603.29025 findings).

---

## Benchmark Results

| Strategy | Accuracy | Avg Reward | Trap Rate |
|----------|----------|------------|-----------|
| Random | 50% | 0.35 | 50% |
| Keyword Match | 47% | 0.39 | 53% |
| Large LLM (70B) | ~68% | ~0.55 | ~32% |
| **Trained on HOA** | **~85%** | **~0.80** | **~15%** |

---

## Research Basis

From arXiv:2603.29025:
- Distance/cost cue is **8.7-38x more influential** than goal constraints
- **No model exceeds 75%** on strict evaluation
- Pattern matching via **keyword association**, not compositional reasoning
- Minimal hints recover **+15 percentage points**
- Goal-decomposition prompting: **+6-9pp** improvement

---

## Team Z

**Vishruth M R** (vishruthmr25@gmail.com)  
**Akshay Kumar** (akshaykumarhudedmani@gmail.com)

OpenEnv Hackathon 2026 — Round 1 Submission

---

MIT License • [arXiv:2603.29025](https://arxiv.org/abs/2603.29025) • [HF Space](https://huggingface.co/spaces/wysh3/heuristic-override-arena)
