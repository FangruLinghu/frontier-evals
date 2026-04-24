# PaperBench project instructions

## Reporting eval runs

After every eval run (single or batch), report a comparison table with these columns:

- Full score (root of the weighted task tree; stored in `grader_output.json`'s `score` field)
- Code-dev score (recompute by pruning the tree to Code Development leaves and re-aggregating with `score_from_children`; equivalent to `TaskNode.code_only()` + `update_all_grades`)
- Passed tasks (leaf pass counts, broken down by task category where relevant)
- Cost (derived from `grader_output.json`'s `token_usage`, priced per model — e.g. o4-mini $1.10/$4.40 per 1M in/out, gpt-4o-2024-08-06 $2.50/$10.00 per 1M in/out)

Include all runs that are being compared in the same table, with prior baselines alongside new runs when relevant.
