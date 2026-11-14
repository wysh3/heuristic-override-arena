"""Reference baselines computed in-process (no network, no LLM, deterministic).

Usage:  uv run python scripts/baselines.py [--runs 200]

Strategies
  oracle        always the correct option + all ground-truth keywords (upper bound)
  correct_only  always the correct option, no explanation text
  trap          always the heuristic trap (what a shortcut-following agent does)
  random        uniform random option, no explanation text
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from hoa_env.grader import grade_scenario
from hoa_env.models import HOAAction

SCEN = Path(__file__).parent.parent / "scenarios"


def load():
    out = []
    for f in sorted(SCEN.glob("*.json")):
        out.extend(json.loads(f.read_text()))
    return out


def action_for(strategy, s, rng):
    gt = s["ground_truth"]
    if strategy == "oracle":
        return HOAAction(
            choice=gt["correct_choice"],
            constraint_identified=" ".join(gt["constraint_keywords"]),
            heuristic_identified=" ".join(gt["heuristic_keywords"]),
        )
    if strategy == "correct_only":
        return HOAAction(choice=gt["correct_choice"])
    if strategy == "trap":
        return HOAAction(choice=gt["trap_choice"])
    return HOAAction(choice=rng.choice(sorted(s["options"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()
    scenarios, rng = load(), random.Random(0)
    print(f"{'strategy':<14}{'accuracy':>10}{'avg reward':>12}{'trap rate':>11}")
    for strat in ["oracle", "correct_only", "random", "trap"]:
        reps = 1 if strat != "random" else args.runs
        acc, rew, trap = [], [], []
        for _ in range(reps):
            r = []
            for s in scenarios:
                a = action_for(strat, s, rng)
                score, _ = grade_scenario(a, s["ground_truth"])
                r.append((score, a.choice == s["ground_truth"]["correct_choice"],
                          a.choice == s["ground_truth"]["trap_choice"]))
            rew.append(statistics.mean(x[0] for x in r))
            acc.append(statistics.mean(x[1] for x in r))
            trap.append(statistics.mean(x[2] for x in r))
        print(f"{strat:<14}{statistics.mean(acc):>10.1%}{statistics.mean(rew):>12.3f}{statistics.mean(trap):>11.1%}")


if __name__ == "__main__":
    main()
