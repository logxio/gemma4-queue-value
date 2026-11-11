"""Descriptive Gemma 4 case study (12 issues x 2 configurations; n = 24).

Where the per-issue budget went (unresolved runs, loops, timeouts), and what the
frozen runtime gate (analysis/gemma_transfer_gate.json) and a loop signature
would have flagged. Only aggregate behavior is written out; no prompt, argument
text or helper design leaves this script. Writes evidence/results/gemma_case_study.json.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(REPO / "analysis"))
from runtime_features import TEST_RE, files  # noqa: E402

DATA = REPO / "evidence" / "gemma_diag12"
GATE = json.loads((REPO / "analysis" / "gemma_transfer_gate.json").read_text())
OUT = REPO / "evidence" / "results" / "gemma_case_study.json"
ARMS = {"A_baseline": ("baseline_reasoning_on", "A_baseline"), "B_graph_before": ("tuned_reasoning_off_with_helper", "B_v2")}
LOOP_RUN = 5
HELPER = "code_analyzer"


def calls_of(trace):
    out = []
    for step in trace["steps"]:
        for call in step.get("tool_calls") or []:
            extra = call.get("extra", {})
            args = call.get("arguments", {})
            text = json.dumps(args, sort_keys=True)
            out.append({"fn": call.get("function_name", ""), "args": text, "near": call.get("function_name", "") + "|" + text[:80],
                        "author": extra.get("author") or step.get("extra", {}).get("author"), "t": extra.get("timestamp", 0.0),
                        "action_text": " ".join(str(v) for v in args.values())})
    return sorted(out, key=lambda c: c["t"])


def runs(seq):
    """Lengths and start indices of maximal runs of equal consecutive items."""
    out, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            out.append((start, i - start))
            start = i
    return out


def gate_score(main_actions):
    first = main_actions[:GATE["checkpoint_actions"]]
    paths = [p for a in first for p in files(a)]
    source = len({p for p in paths if not TEST_RE.search(p)})
    x = (math.log1p(source) - GATE["mean"][0]) / GATE["scale"][0]
    x = max(-5, min(5, x))
    z = x * GATE["weights"][0] + GATE["bias"]
    return source, 1 / (1 + math.exp(-z))


def tie(task, arm):
    """Hash tie-break of the frozen decision rule (runtime_replay.bucket with salt 'tie')."""
    return hashlib.sha256((f"{task}:{arm}" + ":tie").encode()).digest()[0] * 1e-9


def main():
    rows = [json.loads(line) for line in (DATA / "rows.jsonl").read_text().splitlines() if line.strip()]
    per_run = []
    for r in rows:
        label, folder = ARMS[r["arm"]]
        trace = json.loads((DATA / "traces" / folder / f"trace_{r['task_id']}.json").read_text())
        calls = calls_of(trace)
        exact = runs([c["fn"] + c["args"] for c in calls])
        near = runs([c["near"] for c in calls])
        loop_calls = sum(n for _, n in near if n >= LOOP_RUN)
        loop_helper = sum(n for s, n in near if n >= LOOP_RUN and calls[s]["author"] == HELPER)
        main_actions = [c["action_text"] for c in calls if c["author"] != HELPER]
        source, score = gate_score(main_actions)
        eligible = len(main_actions) > GATE["checkpoint_actions"]
        # a run of >= LOOP_RUN starting before k is visible by k only if it has reached LOOP_RUN calls by k
        loop_at = {k: any(s + LOOP_RUN <= k and n >= LOOP_RUN for s, n in near) for k in (10, 20)}
        start = trace["steps"][0].get("extra", {}).get("timestamp", calls[0]["t"] if calls else 0.0)
        freed = {"calls_after_20": max(0, len(calls) - 20),
                 "seconds_after_20th_call": round(max(0.0, r["duration_seconds"] - (calls[19]["t"] - start)), 1) if len(calls) >= 20 else 0.0}
        timed_out = "timeout" in (r.get("error") or "").lower()
        per_run.append({
            "arm": label, "task": r["task_id"], "resolved": bool(r["resolved"]), "timed_out": timed_out,
            "seconds": round(r["duration_seconds"], 1), "tool_calls_logged": r["tool_calls"], "llm_calls": r["llm_calls"],
            "tool_calls_in_trace": len(calls), "helper_calls": sum(c["author"] == HELPER for c in calls),
            "main_agent_calls": len(main_actions),
            "longest_exact_repeat": max((n for _, n in exact), default=0),
            "longest_near_repeat": max((n for _, n in near), default=0),
            "loop_calls": loop_calls, "loop_calls_by_helper": loop_helper,
            "gate_source_paths_at_10": source, "gate_score": round(score, 4),
            "gate_eligible": eligible, "gate_stop": bool(eligible and score + tie(r["task_id"], r["arm"]) < GATE["threshold"]),
            "gate_stop_without_tie_break": bool(eligible and score < GATE["threshold"]),
            "loop_flag_by_10_calls": loop_at[10], "loop_flag_by_20_calls": loop_at[20], **freed,
        })
    arms = {}
    for label, _ in ARMS.values():
        rs = [x for x in per_run if x["arm"] == label]
        tot_calls = sum(x["tool_calls_in_trace"] for x in rs)
        tot_sec = sum(x["seconds"] for x in rs)
        unres = [x for x in rs if not x["resolved"]]
        arms[label] = {
            "runs": len(rs), "resolved": sum(x["resolved"] for x in rs), "timed_out": sum(x["timed_out"] for x in rs),
            "tool_calls": tot_calls, "seconds": round(tot_sec, 1),
            "mean_seconds": round(tot_sec / len(rs), 1), "mean_tool_calls": round(tot_calls / len(rs), 1),
            "share_calls_in_unresolved": round(sum(x["tool_calls_in_trace"] for x in unres) / tot_calls, 3),
            "share_seconds_in_unresolved": round(sum(x["seconds"] for x in unres) / tot_sec, 3),
            "share_seconds_in_timed_out": round(sum(x["seconds"] for x in rs if x["timed_out"]) / tot_sec, 3),
            "share_calls_in_loops": round(sum(x["loop_calls"] for x in rs) / tot_calls, 3),
            "loop_runs": sum(x["loop_calls"] > 0 for x in rs),
            "loop_runs_resolved": sum(x["loop_calls"] > 0 and x["resolved"] for x in rs),
            "gate_stops": sum(x["gate_stop"] for x in rs), "gate_stops_resolved": sum(x["gate_stop"] and x["resolved"] for x in rs),
            "gate_eligible": sum(x["gate_eligible"] for x in rs),
            "loop_flag_10": sum(x["loop_flag_by_10_calls"] for x in rs),
            "loop_flag_10_resolved": sum(x["loop_flag_by_10_calls"] and x["resolved"] for x in rs),
            "loop_flag_20": sum(x["loop_flag_by_20_calls"] for x in rs),
            "loop_flag_20_resolved": sum(x["loop_flag_by_20_calls"] and x["resolved"] for x in rs),
            "gate_stops_without_tie_break": sum(x["gate_stop_without_tie_break"] for x in rs),
            "loop_flag_20_calls_freed": sum(x["calls_after_20"] for x in rs if x["loop_flag_by_20_calls"]),
            "loop_flag_20_seconds_freed": round(sum(x["seconds_after_20th_call"] for x in rs if x["loop_flag_by_20_calls"]), 1),
        }
    tasks = sorted({x["task"] for x in per_run})
    by_task = {t: {x["arm"]: x["resolved"] for x in per_run if x["task"] == t} for t in tasks}
    queue = {"competition_budget_hours": 12, "per_task_cap_minutes": 60,
             "attempts_if_every_attempt_hits_60_min_cap": 12,
             "attempts_at_observed_mean_seconds": {k: round(12 * 3600 / v["mean_seconds"], 1) for k, v in arms.items()},
             "note": "observed runs had a 4-minute cap, so observed means understate uncapped cost"}
    flagged_tasks = sorted({x["task"] for x in per_run if x["loop_flag_by_20_calls"]})
    result = {"n_runs": len(per_run),
              "loop_flagged_issues_solved_by_any_configuration": sum(any(by_task[t].values()) for t in flagged_tasks),
              "loop_flagged_issues": len(flagged_tasks), "gate": {"checkpoint": GATE["checkpoint_actions"], "threshold": GATE["threshold"],
                                               "features": GATE["features"]},
              "loop_definition": f"run of >= {LOOP_RUN} consecutive tool calls with the same function and the same first 80 characters of arguments",
              "arms": arms, "issues_resolved_by": {"both": sum(all(v.values()) for v in by_task.values()),
                                                   "only_tuned": sum(v["tuned_reasoning_off_with_helper"] and not v["baseline_reasoning_on"] for v in by_task.values()),
                                                   "only_baseline": sum(v["baseline_reasoning_on"] and not v["tuned_reasoning_off_with_helper"] for v in by_task.values()),
                                                   "neither": sum(not any(v.values()) for v in by_task.values())},
              "queue": queue, "runs": per_run}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    for x in per_run:
        print(x["arm"][:8], f"{x['task']:14s}", x["resolved"], x["timed_out"], x["seconds"], x["tool_calls_in_trace"], x["helper_calls"],
              "rep", x["longest_exact_repeat"], x["longest_near_repeat"], "loop", x["loop_calls"], x["loop_calls_by_helper"],
              "gate", x["gate_source_paths_at_10"], x["gate_score"], x["gate_eligible"], x["gate_stop"], "lf", x["loop_flag_by_10_calls"], x["loop_flag_by_20_calls"], x["calls_after_20"], x["seconds_after_20th_call"])


if __name__ == "__main__":
    main()
