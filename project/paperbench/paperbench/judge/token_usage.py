from __future__ import annotations

from dataclasses import dataclass

import openai

from paperbench.judge.graded_task_node import GradedTaskNode


@dataclass
class TokenUsage:
    """Tracks billable token usage across different OpenAI models."""

    def __init__(self) -> None:
        self.usage: dict[str, dict[str, int]] = {}

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        requests: int = 0,
        cached_input_tokens: int = 0,
        reasoning_output_tokens: int = 0,
    ) -> None:
        """
        Add usage for one model.

        ``cached_in`` is a subset of ``in`` and ``reasoning_out`` is a subset
        of ``out``. Do not add either field to the totals again when calculating
        cost.
        """

        if model not in self.usage:
            self.usage[model] = {
                "in": 0,
                "out": 0,
                "requests": 0,
                "cached_in": 0,
                "reasoning_out": 0,
            }
        self.usage[model]["in"] += input_tokens
        self.usage[model]["out"] += output_tokens
        self.usage[model]["requests"] += requests
        self.usage[model]["cached_in"] += cached_input_tokens
        self.usage[model]["reasoning_out"] += reasoning_output_tokens

    def add_from_completion(self, model: str, usage: openai.types.CompletionUsage | None) -> None:
        """Add token usage from an OpenAI completion response."""
        if usage is None:
            return

        prompt_details = usage.prompt_tokens_details
        completion_details = usage.completion_tokens_details
        self.add_usage(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            requests=1,
            cached_input_tokens=(
                prompt_details.cached_tokens
                if prompt_details is not None and prompt_details.cached_tokens is not None
                else 0
            ),
            reasoning_output_tokens=(
                completion_details.reasoning_tokens
                if completion_details is not None
                and completion_details.reasoning_tokens is not None
                else 0
            ),
        )

    def merge(self, other: TokenUsage) -> None:
        """Merge another usage accumulator into this one."""

        for model, usage in other.usage.items():
            self.add_usage(
                model,
                usage["in"],
                usage["out"],
                requests=usage.get("requests", 0),
                cached_input_tokens=usage.get("cached_in", 0),
                reasoning_output_tokens=usage.get("reasoning_out", 0),
            )

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Convert usage to a dictionary format."""
        return self.usage

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> TokenUsage:
        """Create a TokenUsage instance from a dictionary."""
        token_usage = cls()
        for model, usage in data.items():
            token_usage.add_usage(
                model,
                usage["in"],
                usage["out"],
                requests=usage.get("requests", 0),
                cached_input_tokens=usage.get("cached_in", 0),
                reasoning_output_tokens=usage.get("reasoning_out", 0),
            )
        return token_usage


def _get_leaf_node_token_usages(task: GradedTaskNode) -> list[TokenUsage]:
    """Recursively extract token usage from leaf nodes of the task tree"""

    if task.is_leaf():
        # need this check because judge_metadata may be malformed in case of node errors
        if task.judge_metadata is not None and task.judge_metadata.get("token_usage"):
            return [TokenUsage.from_dict(task.judge_metadata["token_usage"])]
        else:
            return []

    token_usages = []

    for t in task.sub_tasks:
        t_usages = _get_leaf_node_token_usages(t)
        token_usages.extend(t_usages)
    return token_usages


def get_total_token_usage(graded_task_tree: GradedTaskNode) -> TokenUsage:
    """
    Gets the total token usage summed across all leaf nodes of the task tree
    Assumes the judge_metadata dict a `token_usage` key of type `TokenUsage`
    """
    token_usages = _get_leaf_node_token_usages(graded_task_tree)

    total_token_usage = TokenUsage()
    for token_usage in token_usages:
        total_token_usage.merge(token_usage)

    return total_token_usage
