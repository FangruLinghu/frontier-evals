#!/bin/bash

# Run paperbench judge on existing submission
# Paper: all-in-one
# Submission: submissions/all-in-one
# Output: eval_outputs/

# For dummy judge (testing):
# uv run python paperbench/scripts/run_judge.py \
#     submission_path=submissions/all-in-one \
#     paper_id=all-in-one \
#     judge=dummy \
#     out_dir=eval_outputs/

# For real grading with gpt-4o-mini:
uv run python paperbench/scripts/run_judge.py \
    submission_path=submissions/bridging-data-gaps_v1 \
    paper_id=bridging-data-gaps \
    judge=simple \
    out_dir=outputs/bridging-data-gaps_v1 \
    completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    completer_config.model=gpt-4o-mini
