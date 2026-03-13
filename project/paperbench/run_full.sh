#!/bin/bash

# Run full paperbench pipeline with reproduction
# This runs: submission extraction -> reproduce.sh execution -> grading

# Configuration
SUBMISSION_DIR="submissions"
PAPER_SPLIT="debug"  # Uses experiments/splits/debug.txt (all-in-one)
JUDGE="dummy"        # Use "simple" for real grading, "dummy" for testing

# For testing (dummy judge, no API calls):
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=$PAPER_SPLIT \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$SUBMISSION_DIR \
    "paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig" \
    "paperbench.solver.computer_runtime.env.pull_from_registry=false" \
    paperbench.judge.scaffold=$JUDGE \
    runner.recorder=nanoeval.json_recorder:json_recorder

# For real grading with gpt-4o-mini, uncomment below:
# uv run python -m paperbench.nano.entrypoint \
#     paperbench.paper_split=$PAPER_SPLIT \
#     paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
#     paperbench.solver.submissions_dir=$SUBMISSION_DIR \
#     "paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig" \
#     "paperbench.solver.computer_runtime.env.pull_from_registry=false" \
#     paperbench.judge.scaffold=simple \
#     paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
#     paperbench.judge.completer_config.model=gpt-4o-mini \
#     runner.recorder=nanoeval.json_recorder:json_recorder
