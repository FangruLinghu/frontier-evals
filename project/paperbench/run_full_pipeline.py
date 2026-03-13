#!/usr/bin/env python3
"""
Run full paperbench pipeline with reproduction on a direct submission.
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from alcatraz.clusters.local import LocalConfig
from nanoeval.eval import EvalSpec
from nanoeval.evaluation import run
from nanoeval.setup import global_exit_stack
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    NetworkMode,
)
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
    AlcatrazComputerRuntimeNoJupyter,
)
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)

from paperbench.nano.entrypoint import DefaultRunnerArgs
from paperbench.nano.eval import PaperBench
from paperbench.nano.logging import PaperBenchLibraryConfig, setup_logging
from paperbench.nano.structs import JudgeConfig, ReproductionConfig
from paperbench.solvers.direct_submission.solver import PBDirectSubmissionSolver


async def main():
    # Configuration
    submissions_dir = "submissions_for_run"
    paper_split = "bridging-data-gaps"  # Uses experiments/splits/bridging-data-gaps.txt
    runs_dir = "runs_full"
    docker_image = "pb-env:latest"

    # Skip reproduction for now (set to False to enable)
    skip_reproduction = False

    # Judge type: "dummy" for testing, "simple" for real grading
    judge_scaffold = "simple"

    # Create solver config
    solver_config = PBDirectSubmissionSolver(
        submissions_dir=submissions_dir,
        computer_runtime=AlcatrazComputerRuntime(
            env=LocalConfig(pull_from_registry=False),
        ),
    )

    # Create reproduction config (use NoJupyter for reproducer image)
    reproduction_config = ReproductionConfig(
        timeout=100 * 3600,
        retry_threshold=600,
        skip_reproduction=skip_reproduction,
        computer_runtime=AlcatrazComputerRuntimeNoJupyter(
            env=LocalConfig(pull_from_registry=False),
        ),
        computer_config=ComputerConfiguration(
            docker_image="pb-reproducer:latest",
            network_mode=NetworkMode.UNPROXIED,
        ),
    )

    # Create judge config
    judge_config = JudgeConfig(
        grade=True,
        grade_locally=True,
        scaffold=judge_scaffold,
        completer_config=OpenAICompletionsTurnCompleter.Config(model="gpt-4o-mini"),
        computer_runtime=AlcatrazComputerRuntime(
            env=LocalConfig(pull_from_registry=False),
        ),
        computer_config=ComputerConfiguration(
            docker_image=docker_image,
            network_mode=NetworkMode.UNPROXIED,
        ),
    )

    async with global_exit_stack:
        setup_logging(PaperBenchLibraryConfig())

        paperbench = PaperBench(
            paper_split=paper_split,
            solver=solver_config,
            judge=judge_config,
            reproduction=reproduction_config,
            runs_dir=runs_dir,
            docker_image=docker_image,
        )

        eval_spec = EvalSpec(
            eval=paperbench,
            runner=DefaultRunnerArgs(max_retries=0),
        )

        print(f"Running PaperBench pipeline...")
        print(f"  Paper split: {paper_split}")
        print(f"  Submissions dir: {submissions_dir}")
        print(f"  Skip reproduction: {skip_reproduction}")
        print(f"  Judge: {judge_scaffold}")

        await run(eval_spec)

        print("Pipeline completed. Check runs/ directory for results.")


if __name__ == "__main__":
    asyncio.run(main())
