#!/usr/bin/env python3
"""
Run full paperbench pipeline on TACC Vista using Apptainer + vLLM.

Prerequisites:
  1. vLLM serving on localhost:8000 (or set OPENAI_BASE_URL)
  2. pb-reproducer.sif built from reproducer.Dockerfile
  3. Set OPENAI_BASE_URL and OPENAI_API_KEY env vars

Usage:
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=dummy
  python run_full_pipeline_vista.py
"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from nanoeval.eval import EvalSpec
from nanoeval.evaluation import run
from nanoeval.setup import global_exit_stack
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    NetworkMode,
)
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)

from paperbench.infra.apptainer_interface import ApptainerComputerRuntime
from paperbench.nano.entrypoint import DefaultRunnerArgs
from paperbench.nano.eval import PaperBench
from paperbench.nano.logging import PaperBenchLibraryConfig, setup_logging
from paperbench.nano.structs import JudgeConfig, ReproductionConfig
from paperbench.solvers.direct_submission.solver import PBDirectSubmissionSolver


async def main():
    # ---- Configuration ----
    submissions_dir = "submissions_for_run"
    paper_split = "bridging-data-gaps"
    runs_dir = "runs_full"
    skip_reproduction = False
    judge_scaffold = "simple"

    # Apptainer .sif image path (build with: apptainer build pb-reproducer.sif docker://pb-reproducer:latest)
    reproducer_sif = os.environ.get("PB_REPRODUCER_SIF", "pb-reproducer.sif")

    # vLLM model name (must match --served-model-name in vLLM)
    judge_model = os.environ.get("PB_JUDGE_MODEL", "gpt-4o-mini")

    # GPU access for reproduction
    use_gpu = os.environ.get("PB_USE_GPU", "true").lower() == "true"

    # ---- Solver ----
    solver_config = PBDirectSubmissionSolver(
        submissions_dir=submissions_dir,
        computer_runtime=ApptainerComputerRuntime(
            sif_path=reproducer_sif,
            use_gpu=use_gpu,
        ),
    )

    # ---- Reproduction ----
    reproduction_config = ReproductionConfig(
        timeout=100 * 3600,
        retry_threshold=600,
        skip_reproduction=skip_reproduction,
        computer_runtime=ApptainerComputerRuntime(
            sif_path=reproducer_sif,
            use_gpu=use_gpu,
        ),
        computer_config=ComputerConfiguration(
            docker_image=None,  # Not used with Apptainer
            network_mode=NetworkMode.UNPROXIED,
        ),
    )

    # ---- Judge ----
    judge_config = JudgeConfig(
        grade=True,
        grade_locally=True,
        scaffold=judge_scaffold,
        completer_config=OpenAICompletionsTurnCompleter.Config(model=judge_model),
        computer_runtime=ApptainerComputerRuntime(
            sif_path=reproducer_sif,
            use_gpu=use_gpu,
        ),
        computer_config=ComputerConfiguration(
            docker_image=None,
            network_mode=NetworkMode.UNPROXIED,
        ),
    )

    # ---- Run ----
    async with global_exit_stack:
        setup_logging(PaperBenchLibraryConfig())

        paperbench = PaperBench(
            paper_split=paper_split,
            solver=solver_config,
            judge=judge_config,
            reproduction=reproduction_config,
            runs_dir=runs_dir,
            docker_image="unused",  # Not used with Apptainer
        )

        eval_spec = EvalSpec(
            eval=paperbench,
            runner=DefaultRunnerArgs(max_retries=0),
        )

        print("Running PaperBench pipeline (Vista/Apptainer)...")
        print(f"  Paper split: {paper_split}")
        print(f"  Submissions dir: {submissions_dir}")
        print(f"  Reproducer SIF: {reproducer_sif}")
        print(f"  Judge model: {judge_model}")
        print(f"  Skip reproduction: {skip_reproduction}")
        print(f"  GPU: {use_gpu}")

        await run(eval_spec)

        print("Pipeline completed. Check runs_full/ directory for results.")


if __name__ == "__main__":
    asyncio.run(main())
