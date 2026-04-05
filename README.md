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
  - decision-making
short_description: Train AI to override tempting shortcuts when rules say no
---

# 🧠 Heuristic Override Arena

**An OpenEnv environment where AI agents learn to resist obvious-but-wrong choices.**

> Based on [arXiv:2603.29025](https://arxiv.org/abs/2603.29025) — CMU, March 2026

---

## The Problem

AI agents fall for surface patterns even when explicit rules say otherwise:

| Domain | The Tempting Shortcut | The Actual Rule |
|--------|----------------------|-----------------|
| **Procurement** | "Cheaper vendor" | Missing required certification |
| **HR** | "More experience" | Failed background check |
| **Medical** | "Sicker patient first" | Protocol says wait |

**This environment trains agents to recognize and override these traps.**

---

## How It Works

```python
# Agent sees a scenario
obs = await env.reset(task="procurement")

# Agent must choose AND explain why
action = HOAAction(
    choice="B",
    constraint_identified="HIPAA compliance required",
    heuristic_identified="lower cost"
)

# Environment rewards understanding, not just correct answers
result = await env.step(action)  # reward: 0.80
```

---

## Scoring

```
reward = 0.6×correct + 0.2×constraint_id + 0.2×heuristic_id − 0.3×trap_penalty
```

- **Correct choice alone**: 0.60
- **+ Identified constraint**: 0.80  
- **+ Identified heuristic**: 1.00
- **Fell for trap**: −0.30 penalty

---

## Tasks

| Task | Scenarios | Heuristic Types |
|------|-----------|-----------------|
| `procurement` | 25 | Cost, relationship, speed |
| `hr_decision` | 25 | Experience, performance |
| `medical_triage` | 25 | Severity, urgency, proximity |

---

## Quick Start

```bash
# Test the environment
curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset

# Run inference
export HF_TOKEN="your_token"
python inference.py
```

---

## Team Z

**Vishruth M R** (vishruthmr25@gmail.com)  
**Akshay Kumar** (akshaykumarhudedmani@gmail.com)

OpenEnv Hackathon 2026 — Round 1 Submission

---

MIT License • [arXiv:2603.29025](https://arxiv.org/abs/2603.29025)
