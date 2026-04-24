# Modification Log

## 2026-04-24

### Prepared repo for push to fork
- Added `submissions/`, `outputs/`, and model checkpoint patterns (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`) to `.gitignore` — these are generated/data, not source
- Dropped previously-tracked `outputs/all-in-one/*.jsonl` message traces and old `submissions/**` from the repo

## 2026-04-21

### Added project CLAUDE.md
- Rule: after every eval run, report a comparison table with full score / code-dev score / passed tasks / cost

### Code-dev run: ptc/minimax-m2.7/robust-clip_debug
- Full run → 51.47% (+13.6 over prior baseline); code-only run → 63.92%
- ~13 pt gap between modes explained by `build_judge_task_prompt(code_only)` and `reproduce.sh` attachment differences

## 2026-04-20

### Added SCORE_COMPONENTS.md
- Short explainer of rubric score aggregation and Code Dev / Code Exec / Result Analysis split

### Code-dev-only batch for 3 new m2.7 submissions
- Created `run_code_dev.sh` (loop with `code_only=true`)
- paper2code/robust-clip_v1 30.16%, paper2code/robust-clip_v1_runnable 30.76%, deepcode/robust-clip_full 35.06%
- Per-run cost extracted from `token_usage` in `grader_output.json`

## 2026-04-17

### Full-rubric m2.7 batch across 3 pipelines × 3 papers
- ptc / deepcode / paper2code × adaptive-pruning / bridging-data-gaps / robust-clip
- o4-mini judge, 9 runs, ~$76 total
- ptc wins every paper on both full-rubric and code-dev-only scoring
- Verified score recomputation matches `paperbench`'s own `TaskNode.code_only()` + `update_all_grades()` to 10 decimals

## 2026-04-16

### Apptainer/Vista scaffolding for TACC HPC
- `paperbench/infra/apptainer_interface.py` — `ApptainerComputerInterface` + `ApptainerComputerRuntime` satisfying the `ComputerInterface` / `ComputerRuntime` ABCs via `apptainer exec --bind` with a tempdir-based workdir
- `paperbench/scripts/build-apptainer-images.sh` — convert local Docker image to `.sif`
- `run_full_pipeline_vista.py` — Vista entry point (Apptainer + vLLM local judge via `OPENAI_BASE_URL`)
- `run_vista.slurm` — SLURM job template
- No changes to Alcatraz/Docker paths; existing `reproduce.py`, `computer_utils.py`, grading code work unchanged because they only use `ComputerInterface`

## 2026-04-10

### Evaluated adaptive-pruning codebases (ptc pipeline, 3 models)
- Copied codebases from `ptc_agent_main/data/experiments/adaptive-pruning_{claude,gpt,minimax}` to `submissions/ptc/{claude,gpt,minimax}/adaptive-pruning/codebase/`
- Created `experiments/splits/adaptive-pruning.txt`
- Ran `run_judge.py` with `judge=simple`, `code_only=True`, `completer_config.model=o4-mini` via `uv run`
- Results (86 leaf nodes each):
  - ptc/claude/adaptive-pruning: score=0.2280, tokens 4.47M in / 150K out
  - ptc/gpt/adaptive-pruning: score=0.3964, tokens 2.47M in / 144K out
  - ptc/minimax/adaptive-pruning: score=0.4187, tokens 5.51M in / 143K out

### Evaluated with_gap_agent adaptive-pruning codebase
- Copied from `ptc_agent_main/data/output/adaptive-pruning_with_gap_agent/codebase` to `submissions/ptc/with_gap_agent/adaptive-pruning/codebase/`
- ptc/with_gap_agent/adaptive-pruning: score=0.4810, tokens 5.22M in / 157K out (o4-mini judge, code-only, 86 leaves)

## 2026-04-08

### Added paper_overview.md
- Created `paper_overview.md` with all 23 PaperBench papers categorized by research focus
- Categories: Model Improvement, Continual Learning, Analysis/Interpretability, RL Methods, LLM Reasoning, Statistical Inference, Data Selection
- Includes framework info (PyTorch/TensorFlow), sub-task counts, points, and GitHub repo links
- Reproducibility quick reference for model improvement papers sorted by difficulty

## 2026-03-30

### Reorganized submissions and outputs directories
- Created pipeline-based directory structure: `{pipeline}/{model}/{paper}/codebase/`
- Pipelines: `ptc`, `paper2code`, `deepcode`, `claude_code`
- Archived all old submissions and outputs to `archive/`

### Added robust-clip paper split
- Created `experiments/splits/robust-clip.txt`

### Copied PTC codebases for evaluation
- `submissions/ptc/minimax-m2.5/robust-clip/codebase/` — from ptc_agent output_minimax_m2.5
- `submissions/ptc/qwen3-32b/robust-clip/codebase/` — from ptc_agent output_qwen3_32b

### Updated run_full_pipeline.py for PTC minimax-m2.5 run
- `submissions_dir` → `submissions/ptc/minimax-m2.5`
- `paper_split` → `robust-clip`
- `skip_reproduction` → `True` (code-only)
- Judge model → `gpt-5.1` (was `gpt-4o-mini`)

### Added gpt-5.1 to context window lengths
- `common/preparedness_turn_completer/preparedness_turn_completer/utils.py`
- Added `"gpt-5.1": 400_000` to `CONTEXT_WINDOW_LENGTHS`
