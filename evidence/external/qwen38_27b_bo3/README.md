---
license: cc-by-4.0
language:
- en
tags:
- software-engineering
- swe-rebench
- agent-evaluation
configs:
- config_name: tasks
  default: true
  data_files:
  - split: train
    path: data/tasks-*.parquet
- config_name: rollouts
  data_files:
  - split: train
    path: data/rollouts-*.parquet
---

# SWE-Rebench-V2 Qwen3.8-27B BO3 annotations

This is a compact annotation overlay for `nebius/SWE-rebench-V2` at revision
`10c49ab856fd0e62097815ba5909dfc4f31e7f93`. It summarizes campaign `qwen38-swe-rebench-verified-all-bo3-ctx262k-seed20260822` using model `openai/Qwen/Qwen3.8-27B`.
It does not redistribute gold patches, test patches, raw trajectories, exception
text, credentials, service endpoints, or machine-local paths.

> **Incomplete snapshot — do not report this as final pass@3.** Only 15,209/18,816 expected slots are validly scored; 61/64 shard completion markers are present. Task-level `pass_at_3` is null unless that task has all three valid slots.

## Snapshot accounting

- Tasks: 6,272
- Valid scored rollouts: 15,209/18,816
- Successful rollouts: 9,026
- Fully evaluated BO3 tasks: 5,056/6,272
- Observed-solvable tasks: 3,823
- Strict-solvable tasks: 3,816
- Missing / terminal-error / retryable-error slots: 638 /
  2,969 / 0
- Completed shard markers: 61/64

## Configs

- `tasks`: one row per source task, with BO3 outcomes, recursion counts, compute
  summaries, and an interpretable difficulty tier.
- `rollouts`: one row per expected task/rollout slot, including missing or invalid
  slots. `success` is null when `valid_scored` is false; invalid infrastructure
  records are never silently counted as model failures.

Load either config with:

```python
from datasets import load_dataset

tasks = load_dataset("PATH_OR_REPO", "tasks", split="train")
rollouts = load_dataset("PATH_OR_REPO", "rollouts", split="train")
```

`task_annotations.csv` is included for spreadsheet workflows. Join this overlay
to the pinned source dataset using `instance_id`; `dataset_index` is also retained
to reproduce the exact original catalog ordering.

## Solvability and difficulty

`observed_solvable` means at least one valid rollout received evaluator reward > 0.
`strict_solvable` additionally requires all three rollout slots to be validly
scored. The two corresponding ID lists are included at the bundle root.

Difficulty is deliberately coarse: `solved_3_of_3`, `solved_2_of_3`, and
`solved_1_of_3`. A 0/3 result is `no_success_in_3`, not "unsolvable". Incomplete
tasks are not ranked. `difficulty_rank_hardest_first` sorts complete solved tasks
by fewer successes, recursive-only observed status, greater successful tree-step
cost, and greater observed successful context; it is a harness-specific heuristic,
not an intrinsic or statistically precise measure of task difficulty.

## Recursion semantics

A rollout is `recursive` only when its saved trajectory tree contains at least one
materialized child trajectory. `realized_subagents` is the number of child tree
nodes and `tree_steps` sums steps over the root and all descendants. Recursive-only
successes are descriptive rather than causal because the model chose when to
launch subagents, usually after encountering difficulty.

## Validity semantics

A valid rollout has a numeric terminal reward and no explicit infrastructure/error
or nonterminal status. A configured one-hour trajectory timeout is a valid model
failure and is identified by `timed_out`. Missing, unreadable, unscored, and
infrastructure-failed slots remain in `rollouts` with `valid_scored = false`.

Context columns report the largest persisted per-call token usage observed in the
OpenHands root or rollout tree. They do not estimate the minimum context window
needed to reproduce a success.

## License

The source dataset is CC BY 4.0. Respect the license of each underlying repository;
the source dataset's per-row `license` is retained for convenience.
