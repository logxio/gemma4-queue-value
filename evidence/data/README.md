# Derived per-issue rows from public SWE-bench trajectories

`runtime_rows_{5,10,15}.json`: one row per (system, split, issue) for the checkpoint of 5, 10 or 15 agent actions, produced by `analysis/runtime_features.py` from public trajectories in the SWE-bench experiments index (pinned at commit 40f164d5b8f1d249bf95a6df8b74b577fd8e519d) and official per-instance outcomes. 2,544 rows per file: Verified 465 issues × 4 systems (sweagent, agentless, openhands, mini), Lite 192 issues × 2 systems (sweagent, openhands), Multilingual 300 issues × 1 system (mini).

Fields: `instance_id`, `repo`, `system`, `split`, `resolved` (official outcome), `full` (full-run model calls; OpenHands uses assistant turns), `prefix` (checkpoint), and features from the agent's own outgoing actions up to the checkpoint: `path_mentions`, `unique_paths`, `source_paths`, `test_path_share`, `focus`, `repeat_share`, `revisited`, `edit_agreement`, `has_source_edit`, `test_early`, `edit_early`.

`runtime_replay_summary.json`: the output of `analysis/runtime_replay.py` behind the current draft's results table.

The existing scripts read these files from `.ai/swebench/`; copy them there first (`.ai/` is ignored by git). These rows are derived statistics for private analysis; the raw trajectories are not redistributed.
