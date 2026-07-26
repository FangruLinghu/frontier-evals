import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_STAGES = {"file_selection", "grading", "score_parsing"}
MAIN_MODEL = "o4-mini"
PARSER_MODEL = "gpt-4o-2024-08-06"


def _get_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    if not node["sub_tasks"]:
        return [node]

    leaves = []
    for child in node["sub_tasks"]:
        leaves.extend(_get_leaves(child))
    return leaves


def validate_result(result_path: Path, expected_leaves: int) -> dict[str, Any]:
    data = json.loads(result_path.read_text())
    leaves = _get_leaves(data["graded_task_tree"])

    if data["judge_type"] != "simple":
        raise ValueError(f"Unexpected judge type: {data['judge_type']}")
    if data["completer_config"]["model"] != MAIN_MODEL:
        raise ValueError(f"Unexpected judge model: {data['completer_config']['model']}")
    if len(leaves) != expected_leaves or data["num_leaf_nodes"] != expected_leaves:
        raise ValueError(
            f"Expected {expected_leaves} leaves, found "
            f"{len(leaves)} in tree and {data['num_leaf_nodes']} in summary"
        )
    if data["num_invalid_leaf_nodes"] != 0:
        raise ValueError(f"Found {data['num_invalid_leaf_nodes']} invalid leaves")
    if any(
        leaf["task_category"] != "Code Development" or not leaf["valid_score"] for leaf in leaves
    ):
        raise ValueError("Found a non-Code-Development or invalid leaf")

    stage_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for leaf in leaves:
        metadata = leaf.get("judge_metadata") or {}
        usage_by_stage = metadata.get("token_usage_by_stage") or {}
        if set(usage_by_stage) != EXPECTED_STAGES:
            raise ValueError(
                f"Leaf {leaf['id']} has unexpected usage stages: {sorted(usage_by_stage)}"
            )
        for models in usage_by_stage.values():
            if not models:
                raise ValueError(f"Leaf {leaf['id']} has an empty token-usage stage")
            for model, usage in models.items():
                for metric, value in usage.items():
                    stage_totals[model][metric] += value

    if dict(stage_totals) != data["token_usage"]:
        raise ValueError("Per-stage token usage does not match the top-level total")

    expected_main_requests = expected_leaves * 2
    expected_parser_requests = expected_leaves
    if data["token_usage"][MAIN_MODEL]["requests"] != expected_main_requests:
        raise ValueError(
            f"Expected {expected_main_requests} {MAIN_MODEL} requests, found "
            f"{data['token_usage'][MAIN_MODEL]['requests']}"
        )
    if data["token_usage"][PARSER_MODEL]["requests"] != expected_parser_requests:
        raise ValueError(
            f"Expected {expected_parser_requests} {PARSER_MODEL} requests, found "
            f"{data['token_usage'][PARSER_MODEL]['requests']}"
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=Path)
    parser.add_argument("expected_leaves", type=int)
    args = parser.parse_args()

    data = validate_result(args.result_path, args.expected_leaves)
    print(
        f"VALID score={data['score']:.9f} leaves={data['num_leaf_nodes']} "
        f"invalid={data['num_invalid_leaf_nodes']} "
        f"requests={sum(usage['requests'] for usage in data['token_usage'].values())}"
    )
    print("token_usage=" + json.dumps(data["token_usage"], sort_keys=True))


if __name__ == "__main__":
    main()
