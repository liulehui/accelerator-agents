"""Map JAXBench `evaluate --json` output to Harbor rewards.

Follows the JAXBench paper (arXiv:2607.20466, Table 2): per-benchmark speedups
over the XLA baseline are floored at 1x, and incorrect kernels count as 1x.

reward  = max(speedup_vs_baseline, 1.0) if correct else 1.0
speedup = raw speedup_vs_baseline if correct else 0.0
correct = 1 if the kernel passed the correctness check else 0
fast_1  = 1 if correct and speedup_vs_baseline > 1.0 else 0
"""

import json
import sys


def main(result_path: str, reward_path: str) -> None:
    try:
        with open(result_path) as f:
            result = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Could not parse JAXBench result: {e}")
        result = {}

    correct = result.get("status") == "correct"
    speedup = float(result.get("speedup_vs_baseline") or 0.0) if correct else 0.0

    rewards = {
        "reward": round(max(speedup, 1.0), 4),
        "speedup": round(speedup, 4),
        "correct": int(correct),
        "fast_1": int(correct and speedup > 1.0),
    }
    with open(reward_path, "w") as f:
        json.dump(rewards, f)
    print(f"status={result.get('status')} rewards={rewards}")
    sys.exit(0 if correct else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
