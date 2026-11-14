"""Evaluate an LLM on all 100 scenarios with the real grader (no server needed).

Uses the same system prompt and parsing as inference.py.

    export API_BASE_URL=https://integrate.api.nvidia.com/v1
    export MODEL_NAME=nvidia/nemotron-3-super-120b-a12b
    export HF_TOKEN=<api key>
    uv run python scripts/llm_baseline.py --workers 4 --out results.json
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402  (reuses SYSTEM_PROMPT / call_llm / MODEL_NAME)
from hoa_env.environment import TASK_FILES, EXTENDED_FILES  # noqa: E402
from hoa_env.grader import grade_scenario  # noqa: E402
from hoa_env.models import HOAAction  # noqa: E402

FILES = {**TASK_FILES, **EXTENDED_FILES}


def evaluate(item):
    task, s = item
    public = {k: v for k, v in s.items() if k != "ground_truth"}
    out = inference.call_llm(public) or {}
    action = HOAAction(
        choice=str(out.get("choice", "")),
        constraint_identified=str(out.get("constraint_identified", "")),
        heuristic_identified=str(out.get("heuristic_identified", "")),
    )
    gt = s["ground_truth"]
    score, _ = grade_scenario(action, gt)
    return {
        "task": task,
        "id": s["id"],
        "choice": action.choice,
        "correct": action.choice.strip().upper() == gt["correct_choice"].upper(),
        "trap": action.choice.strip().upper() == gt["trap_choice"].upper(),
        "reward": score,
        "error": out.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items = []
    for task, fname in FILES.items():
        for s in json.loads((ROOT / "scenarios" / fname).read_text()):
            items.append((task, s))

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(evaluate, items))

    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)

    print(f"model: {inference.MODEL_NAME}")
    print(f"{'task':<18}{'n':>4}{'accuracy':>10}{'reward':>9}{'trap':>8}{'errors':>8}")
    for task in list(FILES) + ["overall"]:
        rs = results if task == "overall" else by_task[task]
        print(
            f"{task:<18}{len(rs):>4}"
            f"{statistics.mean(r['correct'] for r in rs):>10.1%}"
            f"{statistics.mean(r['reward'] for r in rs):>9.3f}"
            f"{statistics.mean(r['trap'] for r in rs):>8.1%}"
            f"{sum(1 for r in rs if r['error']):>8}"
        )
    n_err = sum(1 for r in results if r["error"])
    if n_err:
        print(f"WARNING: {n_err} LLM calls failed and were scored as the fallback answer 'A'")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
