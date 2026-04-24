# PaperBench Rubric Score Components

A rubric is a weighted, hierarchical tree of requirements: every leaf has a `task_category` of Code Development (is the requirement implemented in code?), Code Execution (does it run successfully?), or Result Analysis (do the produced outputs agree with the paper's reported results?), with internal nodes aggregating leaf scores by `weight`.

Each leaf is graded 0 or 1 (or 0–1 continuous) by the judge against its `requirements` text using the category-specific question. A leaf may also carry a `finegrained_task_category` such as *Method Implementation*, *Experimental Setup*, or *Dataset and Model Acquisition* for sub-axis reporting.

The code-only evaluation mode prunes the tree to Code Development leaves only via `TaskNode.code_only()` (`paperbench/rubric/tasks.py:338`), which is why `bridging-data-gaps_v1` reported 11/172 with the code-dev judge while the full pipeline additionally scores the Code Execution and Result Analysis leaves from `reproduce.log` and produced artifacts.
