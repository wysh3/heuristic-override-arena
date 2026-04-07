"""
Inference script for Heuristic Override Arena.
Runs one episode per task across all 5 domains.
"""
import asyncio
import json
import os
import sys
from typing import List

from openai import OpenAI

# ─── Mandatory env vars ───────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "heuristic-override-arena")
BENCHMARK        = "heuristic-override-arena"
SUCCESS_THRESHOLD = 0.5

TASKS = ["procurement", "hr_decision", "medical_triage", "cognitive_biases", "edge_cases"]

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert decision-maker trained to identify when obvious choices violate hidden constraints.

For each scenario, you must:
1. Identify the surface heuristic (the obvious-seeming choice)
2. Identify the actual constraint that may override it
3. Make the correct choice based on the constraint, not the heuristic

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "choice": "<exact option key from the scenario, e.g. 'A' or 'B'>",
  "constraint_identified": "<the specific rule, policy, certification, or requirement that determines the answer>",
  "heuristic_identified": "<the misleading shortcut you had to override, e.g. 'lower cost', 'more experience'>",
  "reasoning": "<brief explanation of why the constraint overrides the heuristic. Escape all quotes properly>"
}

CRITICAL: Always read the full scenario before choosing. The obvious answer is often wrong."""


def call_llm(scenario: dict) -> dict:
    # Remove ground_truth if somehow present (safety measure)
    safe_scenario = {k: v for k, v in scenario.items() if k != "ground_truth"}
    
    for attempt in range(2):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Scenario:\n{json.dumps(safe_scenario, indent=2)}"},
                ],
                temperature=0.1 + (attempt * 0.2), # slightly more creative on retry
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                for part in raw.split("```"):
                    part = part.strip().lstrip("json").strip()
                    try:
                        return json.loads(part)
                    except Exception:
                        continue
            return json.loads(raw)
        except Exception as e:
            if attempt < 1:
                print(f"[DEBUG] LLM parse error, retrying: {e}", file=sys.stderr, flush=True)
                continue
            print(f"[DEBUG] LLM or parsing error after retry: {e}", file=sys.stderr, flush=True)
            return {"choice": "B", "error": str(e)[:100]}


def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action_dict: dict, reward: float, done: bool, error=None):
    # Compact JSON — no spaces — mandatory
    action_str = json.dumps(action_dict, separators=(",", ":"))
    err_str = str(error)[:100] if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={'true' if done else 'false'} error={err_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


async def run_task(task_name: str) -> float:
    from hoa_env import HOAAction, HOAEnv

    log_start(task_name, MODEL_NAME)

    rewards: List[float] = []
    last_error = None
    step = 0
    action_dict = {}

    try:
        # Connect to environment
        space_url = os.getenv("SPACE_URL", "")
        if space_url:
            env_client = HOAEnv(base_url=space_url)
        else:
            from openenv.core.containers.runtime import LocalDockerProvider
            provider = LocalDockerProvider()
            env_client = await HOAEnv.from_docker_image(LOCAL_IMAGE_NAME, provider=provider, container_port=7860)

        # We will loop exactly until step == 3 or done=True
        env = await env_client.__aenter__()
        result = await env.reset(seed=42, task=task_name)

        while result and not result.done and step < 3:
            step += 1
            if isinstance(result.observation, dict):
                scenario = result.observation.get("scenario", {})
            else:
                scenario = getattr(result.observation, "scenario", {}) or {}
            
            # Generate LLM Action
            try:
                action_dict = call_llm(scenario)
                action = HOAAction(
                    choice=action_dict.get("choice", ""),
                    constraint_identified=action_dict.get("constraint_identified", ""),
                    heuristic_identified=action_dict.get("heuristic_identified", ""),
                    reasoning=action_dict.get("reasoning", ""),
                )
                last_error = None
            except Exception as e:
                action = HOAAction(choice="A", constraint_identified="", heuristic_identified="")
                action_dict = {"choice": "A", "error": "parse_failed"}
                last_error = str(e)[:100]

            # Step Environment
            try:
                result = await env.step(action)
                reward = result.reward if result.reward is not None else 0.0
                rewards.append(reward)
                log_step(step, action_dict, reward, result.done or (step == 3), last_error)
            except Exception as e:
                err_str = str(e)
                print(f"[DEBUG] Network error during step {step}: {err_str}", file=sys.stderr, flush=True)
                
                if "1011" in err_str or "keepalive" in err_str.lower() or "close" in err_str.lower():
                    print("[DEBUG] WebSocket dropped, reconnecting transparently...", file=sys.stderr, flush=True)
                    try:
                        # Close old
                        await env_client.__aexit__(None, None, None)
                    except Exception:
                        pass
                        
                    # Rebuild client and restart env silently to grab next scenario
                    env_client = HOAEnv(base_url=space_url)
                    env = await env_client.__aenter__()
                    result = await env.reset(seed=step*100, task=task_name)
                    
                    # Apply action to new connection
                    result = await env.step(action)
                    reward = result.reward if result.reward is not None else 0.0
                    rewards.append(reward)
                    log_step(step, action_dict, reward, result.done or (step == 3), last_error)
                else:
                    last_error = str(e)[:100]
                    rewards.append(0.0)
                    log_step(step, action_dict, 0.0, True, last_error)
                    break

        await env_client.__aexit__(None, None, None)

    except Exception as e:
        last_error = str(e)[:100]
        if not rewards:
            rewards = [0.0]
        log_step(step + 1, action_dict, 0.0, True, last_error)

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    success = avg >= SUCCESS_THRESHOLD and last_error is None
    log_end(success, step, avg, rewards)
    return avg


async def main():
    all_scores = []
    for task in TASKS:
        score = await run_task(task)
        all_scores.append(score)
        print(f"[DEBUG] # Task {task}: score={score:.4f}", file=sys.stderr, flush=True)
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] # Overall: {overall:.4f}", file=sys.stderr, flush=True)
    sys.exit(0 if all(s >= SUCCESS_THRESHOLD for s in all_scores) else 1)


if __name__ == "__main__":
    asyncio.run(main())
