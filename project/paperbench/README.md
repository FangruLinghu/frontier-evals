# PaperBench Evaluation

## Code-Dev Only 

Grades submission code quality:

```bash
# set submission_path, paper_id and output_dir 
bash run.sh
```

Results saved to `outputs/<name>/grader_output.json`.

## Full Pipeline (with Docker reproduction)

Runs `reproduce.sh` inside Docker, then grades:

```bash
# Build Docker images
bash paperbench/scripts/build-docker-images.sh

# Place submission in submissions_for_run/<paper>/submission/
# Then run:
uv run python run_full_pipeline.py
```

Configure `run_full_pipeline.py` to set paper, judge type, and skip options.

Results saved to `runs_full/<timestamp>/grade.json`.

