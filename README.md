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

**An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that tests whether an AI agent follows an explicit rule when a tempting shortcut points the other way.**

Each scenario is a short business or clinical decision (pick a vendor, a candidate, a patient...) with one binding policy constraint and one surface cue — cheaper, faster, more experienced, more urgent — that points to the *wrong* option. Grading is fully deterministic: no LLM judge, pure Python.

> Motivated by the failure mode described in [arXiv:2603.29025](https://arxiv.org/abs/2603.29025): language models often follow salient surface cues over explicit constraints.

**Live demo:** [wysh3-heuristic-override-arena.hf.space](https://wysh3-heuristic-override-arena.hf.space) · **Space:** [huggingface.co/spaces/wysh3/heuristic-override-arena](https://huggingface.co/spaces/wysh3/heuristic-override-arena) · **Code:** [github.com/wysh3/heuristic-override-arena](https://github.com/wysh3/heuristic-override-arena)

---

## Example

```text
Context:  You are a procurement manager at a healthcare company. Policy mandates
          that all data storage vendors be HIPAA compliant.
Options:  A) SecureVault — $650/month, 99.7% uptime, HIPAA compliant and audited
          B) StorageMax  — $400/month, 99.9% uptime, not HIPAA compliant
Hint:     StorageMax is cheaper and has better uptime        <- the tempting heuristic

Correct:  A  (the constraint decides; price and uptime are irrelevant)
```

## What is in the box

| | |
|---|---|
| **Scenarios** | 100 hand-written decisions across 5 tasks |
| **Tasks** | `procurement` (25, easy) · `hr_decision` (25, medium) · `medical_triage` (25, hard) · `cognitive_biases` (15, medium) · `edge_cases` (10, hard) |
| **Heuristic types** | 22 (severity 17, speed 15, experience 15, cost 12, performance 7, urgency 6, affinity 4, ... see `/tasks`) |
| **Grading** | Deterministic keyword + choice grader with synonym expansion (`hoa_env/grader.py`) |
| **Episodes** | Multi-step: 3 scenarios per episode, deduplicated within an episode |
| **Modes** | single task · curriculum by `difficulty` · full 3-domain progression |
| **Interfaces** | WebSocket (typed Python client) and plain HTTP |

## Reward

Per scenario, clipped to `[0, 1]`:

```
reward = 0.6·correct_choice + 0.2·constraint_match + 0.2·heuristic_match − 0.3·fell_for_trap
```

* `constraint_match` / `heuristic_match` = fraction of the scenario's ground-truth keywords (or known synonyms) found in the first 400 characters of the agent's `constraint_identified` / `heuristic_identified` text. A correct choice with no explanation scores **0.6**; a correct choice that names every keyword scores **1.0**.
* Explanation credit is only given with the correct choice, and long keyword dumps are truncated, so stuffing keywords cannot rescue a wrong or random answer (a random choice with a keyword dump scores about 0.35, versus 0.30 for a random choice with no explanation).
* `fell_for_trap` is 1 if the agent chose the designated trap option. A wrong answer therefore scores 0.
* There is no per-step penalty; the reward does not depend on the step index.
* The observation includes the scenario's `heuristic_signal` (a one-line description of the tempting cue). This is a deliberate hint: it makes the task about *overriding* the cue rather than discovering it.

Episode `done` is true after 3 scenarios (single-task / curriculum modes) or after the 3-domain progression (`procurement → hr_decision → medical_triage`) when no task is given. Calling `step` after `done` starts a new episode.

## Reference baselines

Computed in-process, deterministic, no LLM — reproduce with `uv run python scripts/baselines.py`:

| Strategy | Accuracy | Avg reward | Trap rate |
|---|---|---|---|
| Oracle (correct option + all keywords) | 100% | 1.000 | 0% |
| Correct option, no explanation | 100% | 0.600 | 0% |
| Random option | ~50% | ~0.30 | ~50% |
| Always the trap | 0% | 0.000 | 100% |

### LLM baseline

`nvidia/nemotron-3-ultra-550b-a55b` via the NVIDIA API, zero-shot, temperature 0.1, using the prompt in `inference.py`. 3 runs over **all 100 scenarios** with the current grader (`scripts/llm_baseline.py`), measured 2026-11-15:

| Task | Accuracy | Avg reward (± sd over runs) | Trap rate |
|---|---|---|---|
| procurement | 100% | 0.834 ± 0.009 | 0% |
| hr_decision | 100% | 0.834 ± 0.006 | 0% |
| medical_triage | 93.3% | 0.775 ± 0.025 | 6.7% |
| cognitive_biases | 100% | 0.795 ± 0.005 | 0% |
| edge_cases | 100% | 0.837 ± 0.011 | 0% |
| **Overall** | **98.3%** | **0.814 ± 0.009** | **1.7%** |

All misses came from API calls that returned no answer (9 of 300) and were scored as the fallback answer `A`. **On every question it actually answered, the model chose correctly.** The remaining reward gap (about 0.19) is explanation credit: the model's wording didn't contain all of the expected keywords. An end-to-end `inference.py` run against the live Space with the same model scored 0.82.

**What this means:** with the `heuristic_signal` hint in the observation, a frontier-size model is at ceiling on choice accuracy. The environment separates models on trap avoidance only for smaller models; for large models it measures explanation quality. See *Limitations*.

No trained-model results are reported here.

---

## Quick start

### Python client (WebSocket, recommended)

```python
import asyncio
from hoa_env import HOAAction, HOAEnv

async def main():
    async with HOAEnv(base_url="https://wysh3-heuristic-override-arena.hf.space") as env:
        result = await env.reset(task="procurement")      # or difficulty="easy"
        while not result.done:
            scenario = result.observation.scenario        # context, question, options, heuristic_signal
            result = await env.step(HOAAction(
                choice="A",
                constraint_identified="HIPAA compliance required",
                heuristic_identified="lower cost",
                reasoning="optional, not graded",
            ))
            print(result.reward, result.observation.feedback)

asyncio.run(main())
```

### HTTP

HTTP requests are served by stateless workers, so `/step` continues **the most recently reset episode on the server**. Use the WebSocket client if several users share one server.

```bash
curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task": "procurement"}'

curl -X POST https://wysh3-heuristic-override-arena.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"choice": "A", "constraint_identified": "HIPAA", "heuristic_identified": "cost"}}'
```

`/step` needs the `{"action": {...}}` wrapper; the WebSocket client adds it for you.

### Run an LLM against it

```bash
export SPACE_URL=https://wysh3-heuristic-override-arena.hf.space   # or a local server
export API_BASE_URL=https://router.huggingface.co/v1                # any OpenAI-compatible API
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct   # or nvidia/nemotron-3-ultra-550b-a55b with the NVIDIA API
export HF_TOKEN=...                                                  # key for that API
uv run python inference.py
```

It runs one episode per task and prints `[START]`, `[STEP]` and `[END]` lines.

### Run the server locally

```bash
uv sync
uv run uvicorn hoa_env.server.app:app --port 7860
# or: docker build -t heuristic-override-arena . && docker run -p 7860:7860 heuristic-override-arena
```

### Tests

```bash
uv run pytest        # 128 tests: grader, environment, HTTP bridge, API, scenario data integrity
uv run openenv validate .
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start an episode. Optional body: `task`, `difficulty`, `seed`, `episode_id` |
| `/step` | POST | Submit `{"action": {...}}` |
| `/state` | GET | Current episode state |
| `/info` | GET | Environment metadata (counts computed from the data) |
| `/tasks` | GET | Tasks with scenario counts and heuristic types |
| `/health` | GET | Health check |
| `/` | GET | Demo page |
| `/ws` | WebSocket | Session-isolated episodes (used by the client) |

Action: `choice`, `constraint_identified`, `heuristic_identified`, `reasoning`. Observation: `task_type`, `scenario` (ground truth hidden), `feedback`, `cumulative_reward`, `episode_score`, `done`, `reward`.

## Project layout

```
hoa_env/            environment, grader, models, client, FastAPI app
scenarios/          the 100 scenarios (JSON)
scripts/baselines.py     deterministic reference baselines
scripts/llm_baseline.py  LLM evaluation over all 100 scenarios
inference.py        LLM runner
tests/              pytest suite
openenv.yaml        OpenEnv manifest
Dockerfile          Space / container image
```

## Adding scenarios

Append an object to a file in `scenarios/` with `id`, `context`, `question`, `options`, `heuristic_signal`, `trap_intensity`, and a `ground_truth` containing `correct_choice`, `trap_choice`, `heuristic_type`, `constraint_keywords`, `heuristic_keywords`, `explanation`. `tests/test_scenarios.py` validates the shape.

## Limitations

* Keyword grading rewards naming the expected terms; a correct explanation phrased with unlisted words earns less partial credit (the choice itself is graded exactly).
* Scenarios are short and synthetic; they probe constraint-following, not domain expertise, and are not legal or medical advice.
* Options are binary (A/B).
* Ceiling effect: a 550B model answers every scenario correctly when the API responds (see *LLM baseline*). For a harder benchmark, hide `heuristic_signal` from the observation and add more than two options.

## Team Z

**Vishruth M R** · **Akshay Kumar** — OpenEnv Hackathon 2026

MIT License · [arXiv:2603.29025](https://arxiv.org/abs/2603.29025)
