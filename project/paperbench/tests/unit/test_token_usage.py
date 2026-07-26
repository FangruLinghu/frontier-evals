from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails

from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.token_usage import TokenUsage, get_total_token_usage


def test_token_usage_preserves_billable_details() -> None:
    usage = CompletionUsage(
        completion_tokens=25,
        prompt_tokens=100,
        total_tokens=125,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=40),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=20),
    )
    token_usage = TokenUsage()

    token_usage.add_from_completion("o4-mini", usage)

    assert token_usage.to_dict() == {
        "o4-mini": {
            "in": 100,
            "out": 25,
            "requests": 1,
            "cached_in": 40,
            "reasoning_out": 20,
        }
    }


def test_token_usage_reads_legacy_results() -> None:
    token_usage = TokenUsage.from_dict({"o4-mini": {"in": 100, "out": 25}})

    assert token_usage.to_dict() == {
        "o4-mini": {
            "in": 100,
            "out": 25,
            "requests": 0,
            "cached_in": 0,
            "reasoning_out": 0,
        }
    }


def test_total_token_usage_merges_all_leaf_metrics() -> None:
    first_leaf = GradedTaskNode(
        id="first",
        requirements="First",
        weight=1,
        task_category="Code Development",
        score=1,
        valid_score=True,
        judge_metadata={
            "token_usage": {
                "o4-mini": {
                    "in": 100,
                    "out": 25,
                    "requests": 2,
                    "cached_in": 40,
                    "reasoning_out": 20,
                }
            }
        },
    )
    second_leaf = GradedTaskNode(
        id="second",
        requirements="Second",
        weight=1,
        task_category="Code Development",
        score=1,
        valid_score=True,
        judge_metadata={
            "token_usage": {
                "o4-mini": {
                    "in": 80,
                    "out": 15,
                    "requests": 1,
                    "cached_in": 20,
                    "reasoning_out": 10,
                }
            }
        },
    )
    root = GradedTaskNode(
        id="root",
        requirements="Root",
        weight=1,
        score=1,
        valid_score=True,
        sub_tasks=[first_leaf, second_leaf],
    )

    assert get_total_token_usage(root).to_dict() == {
        "o4-mini": {
            "in": 180,
            "out": 40,
            "requests": 3,
            "cached_in": 60,
            "reasoning_out": 30,
        }
    }
