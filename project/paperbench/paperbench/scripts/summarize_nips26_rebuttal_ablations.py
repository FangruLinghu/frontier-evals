import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from paperbench.scripts.validate_codedev_output import validate_result

PAPERS = {
    "dpo-misspecification": 77,
    "speculative-actions": 150,
    "polar-express": 193,
    "llms-get-lost": 201,
    "crafter-paper": 206,
}
VARIANTS = {
    "gaps_with_debug": "Claude-sonnet4.6",
    "gaps_no_debug": "Claude-sonnet4.6-gaps-no-debug",
    "no_gaps_no_debug": "Claude-sonnet4.6-no-gaps-no-debug",
    "no_gaps_with_debug": "Claude-sonnet4.6-no-gaps-with-debug",
}


def _add_usage(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    for model, usage in source.items():
        if model not in target:
            target[model] = defaultdict(int)
        for metric, value in usage.items():
            target[model][metric] += value


def build_summary(output_root: Path) -> dict[str, Any]:
    aggregate_usage: dict[str, dict[str, int]] = {}
    variant_usage: dict[str, dict[str, dict[str, int]]] = {setting: {} for setting in VARIANTS}
    variant_scores: dict[str, list[float]] = defaultdict(list)
    runs = []

    for paper_id, expected_leaves in PAPERS.items():
        for setting, variant_dir in VARIANTS.items():
            result_path = output_root / variant_dir / paper_id / "grader_output.json"
            data = validate_result(result_path, expected_leaves)
            token_usage = data["token_usage"]
            _add_usage(aggregate_usage, token_usage)
            _add_usage(variant_usage[setting], token_usage)
            variant_scores[setting].append(data["score"])
            runs.append(
                {
                    "paper_id": paper_id,
                    "setting": setting,
                    "variant_dir": variant_dir,
                    "score": data["score"],
                    "num_leaf_nodes": data["num_leaf_nodes"],
                    "num_invalid_leaf_nodes": data["num_invalid_leaf_nodes"],
                    "token_usage": token_usage,
                    "result_path": str(result_path),
                }
            )

    total_leaves = sum(run["num_leaf_nodes"] for run in runs)
    total_requests = sum(usage["requests"] for run in runs for usage in run["token_usage"].values())
    return {
        "judge": {
            "type": "simple",
            "main_model": "o4-mini",
            "parser_model": "gpt-4o-2024-08-06",
            "code_only": True,
        },
        "num_runs": len(runs),
        "total_leaf_nodes": total_leaves,
        "total_requests": total_requests,
        "macro_mean_score_by_setting": {
            setting: sum(scores) / len(scores) for setting, scores in variant_scores.items()
        },
        "aggregate_token_usage": aggregate_usage,
        "aggregate_uncached_input": {
            model: usage["in"] - usage["cached_in"] for model, usage in aggregate_usage.items()
        },
        "token_usage_by_setting": variant_usage,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/code-dev-o4-mini"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/code-dev-o4-mini/nips26-rebuttal-ablation-summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.output_root)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.summary_path}")
    print(
        f"runs={summary['num_runs']} leaves={summary['total_leaf_nodes']} "
        f"requests={summary['total_requests']}"
    )


if __name__ == "__main__":
    main()
