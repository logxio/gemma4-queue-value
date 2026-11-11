"""Gemma 4 repeated rollouts on the competition's public development issues.

Step 1 (derive): read a fetched run directory (task_results.jsonl and traces per
round) and write a compact per-attempt table to evidence/gemma_a100/<run>.jsonl.
Only outcomes, costs, loop signatures and gate fields leave this step; no prompt
or argument text is written.

Step 2 (analyze): the same quantities as analysis/restart_value.py and
analysis/schedule_frontier.py, on Gemma's own rollouts: p, q, q/p, ICC, restart
versus fresh yield, a breadth-versus-depth frontier, and restart value of
loop-flagged and gate-flagged failures. Writes evidence/results/gemma_rollouts_<run>.json.

  python3 analysis/gemma_rollouts.py derive RUN_DIR RUN_NAME
  python3 analysis/gemma_rollouts.py analyze RUN_NAME
"""
import json
import random
import sys
from pathlib import Path

ARGV = list(sys.argv)  # gemma_case_study truncates sys.argv on import
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import gemma_case_study as cs  # noqa: E402

TABLE_DIR = REPO / "evidence" / "gemma_a100"
OUT_DIR = REPO / "evidence" / "results"
SEED = 20260929


def derive(run_dir: Path, name: str) -> None:
    rows = []
    for rdir in sorted(p for p in (run_dir / "results").iterdir() if p.is_dir() and p.name.startswith("r")):
        k = int(rdir.name[1:])
        lines = [(f.parent, line) for f in sorted(rdir.glob("task_results.jsonl")) + sorted(rdir.glob("s*/task_results.jsonl"))
                 for line in f.read_text().splitlines()]
        for base, line in lines:
            if not line.strip():
                continue
            t = json.loads(line)
            task = t["instance_id"]
            trace_path = base / "traces" / f"trace_{task}.json"
            calls = cs.calls_of(json.loads(trace_path.read_text())) if trace_path.exists() else []
            near = cs.runs([c["near"] for c in calls])
            main_actions = [c["action_text"] for c in calls if c["author"] != cs.HELPER]
            source, score = cs.gate_score(main_actions)
            eligible = len(main_actions) > cs.GATE["checkpoint_actions"]
            error = (t.get("error_message") or t.get("error") or "")
            low = error.lower()
            kind = ("timeout" if "session timeout" in low else "context_window" if "contextwindowexceeded" in low
                    else "tool_budget" if "tool call budget" in low else "test_patch_conflict" if "test_patch" in low
                    else "other_error" if error else "none")
            rows.append({
                "task": task, "repo": task.rsplit("_", 1)[0], "round": k, "resolved": bool(t.get("resolved")),
                "seconds": round(float(t.get("duration_seconds") or 0.0), 1),
                "tool_calls": t.get("tool_calls"), "llm_calls": t.get("total_llm_calls"),
                "timed_out": kind == "timeout", "failure_kind": kind,
                # every observed error is the agent's own failure mode, counted as a failure as in the competition;
                # only errors outside these kinds would mark an attempt as not a valid outcome
                "errored": kind == "other_error",
                "trace_found": trace_path.exists(), "calls_in_trace": len(calls),
                "loop_calls": sum(n for _, n in near if n >= cs.LOOP_RUN),
                "loop_flag_by_20_calls": any(s + cs.LOOP_RUN <= 20 and n >= cs.LOOP_RUN for s, n in near),
                "gate_source_paths_at_10": source, "gate_score": round(score, 4), "gate_eligible": eligible,
                "gate_stop": bool(eligible and score + cs.tie(task, f"r{k}") < cs.GATE["threshold"]),
            })
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} attempts -> {out}")


def pct(xs, lo=2.5, hi=97.5):
    xs = sorted(xs)
    return [round(xs[int(len(xs) * lo / 100)], 4), round(xs[min(len(xs) - 1, int(len(xs) * hi / 100))], 4)]


def stats(tasks):
    """tasks: list of lists of (success, calls, seconds) per rollout."""
    att = [a for t in tasks for a in t]
    p = sum(a[0] for a in att) / len(att)
    num = den = cost_calls = cost_sec = 0.0
    for t in tasks:
        for i, a in enumerate(t):
            if a[0]:
                continue
            for j, b in enumerate(t):
                if i != j:
                    den += 1; num += b[0]; cost_calls += b[1]; cost_sec += b[2]
    q = num / den if den else float("nan")
    fresh_calls = sum(a[0] for a in att) / sum(a[1] for a in att)
    fresh_sec = sum(a[0] for a in att) / sum(a[2] for a in att)
    k = len(tasks[0])
    means = [sum(a[0] for a in t) / k for t in tasks]
    msb = k * sum((m - p) ** 2 for m in means) / (len(tasks) - 1)
    msw = sum((a[0] - means[i]) ** 2 for i, t in enumerate(tasks) for a in t) / (len(tasks) * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) else float("nan")
    return {"p": p, "q": q, "q_over_p": q / p if p else float("nan"), "icc": icc,
            "fresh_per_1000_calls": 1000 * fresh_calls, "restart_per_1000_calls": 1000 * num / cost_calls if cost_calls else float("nan"),
            "restart_over_fresh_calls": (num / cost_calls) / fresh_calls if cost_calls and fresh_calls else float("nan"),
            "restart_over_fresh_seconds": (num / cost_sec) / fresh_sec if cost_sec and fresh_sec else float("nan"),
            "failure_anchored_pairs": den}


def frontier(tasks, fractions, reps, rng):
    total = sum(a[2] for t in tasks for a in t)
    out = {f: {"breadth": [], "depth": []} for f in fractions}
    for _ in range(reps):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        order = [rng.sample(t, len(t)) for t in sample]
        for f in fractions:
            budget = f * total
            # depth-first: retry a task until success or three attempts, then the next task
            spent, solved = 0.0, 0
            for t in order:
                stop = False
                for a in t:
                    if spent + a[2] > budget:
                        stop = True; break
                    spent += a[2]
                    if a[0]:
                        solved += 1; break
                if stop:
                    break
            out[f]["depth"].append(solved)
            # breadth-first: one attempt per task, then second attempts on unsolved, then third
            spent, done, solved, halt = 0.0, set(), 0, False
            for r in range(len(order[0])):
                for i, t in enumerate(order):
                    if i in done:
                        continue
                    if spent + t[r][2] > budget:
                        halt = True; break
                    spent += t[r][2]
                    if t[r][0]:
                        solved += 1; done.add(i)
                if halt:
                    break
            out[f]["breadth"].append(solved)
    return {str(f): {"depth_mean": sum(v["depth"]) / reps, "breadth_mean": sum(v["breadth"]) / reps,
                     "diff_mean": sum(b - d for b, d in zip(v["breadth"], v["depth"])) / reps,
                     "diff_interval": pct([b - d for b, d in zip(v["breadth"], v["depth"])])} for f, v in out.items()}


def targeted(complete, fractions, reps, rng):
    """Breadth-first, then restarts; restarts of loop-flagged failures go first (the flag is visible by call 20)."""
    vals = list(complete.values())
    total = sum(a["seconds"] for v in vals for a in v)
    res = {f: [] for f in fractions}
    base = {f: [] for f in fractions}
    for _ in range(reps):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        order = [rng.sample(v, len(v)) for v in sample]
        for f in fractions:
            budget = f * total
            for mode, sink in (("plain", base), ("loops_first", res)):
                spent, solved, done, pending = 0.0, 0, set(), []
                halt = False
                for i, v in enumerate(order):
                    a = v[0]
                    if spent + a["seconds"] > budget:
                        halt = True; break
                    spent += a["seconds"]
                    if a["resolved"]:
                        solved += 1; done.add(i)
                    else:
                        pending.append((0 if (mode == "loops_first" and a["loop_flag_by_20_calls"]) else 1, i))
                if not halt:
                    for r in range(1, len(order[0])):
                        queue = sorted(pending) if mode == "loops_first" else pending
                        nxt = []
                        for _, i in queue:
                            if i in done:
                                continue
                            a = order[i][r]
                            if spent + a["seconds"] > budget:
                                halt = True; break
                            spent += a["seconds"]
                            if a["resolved"]:
                                solved += 1; done.add(i)
                            else:
                                nxt.append((0 if (mode == "loops_first" and a["loop_flag_by_20_calls"]) else 1, i))
                        if halt:
                            break
                        pending = nxt
                sink[f].append(solved)
    return {str(f): {"breadth_mean": sum(base[f]) / reps, "loops_first_mean": sum(res[f]) / reps,
                     "diff_mean": sum(x - y for x, y in zip(res[f], base[f])) / reps,
                     "diff_interval": pct([x - y for x, y in zip(res[f], base[f])])} for f in fractions}


def analyze(name: str) -> None:
    rows = [json.loads(line) for line in (TABLE_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["task"], {})[r["round"]] = r
    rounds = sorted({r["round"] for r in rows})
    complete = {t: [v[k] for k in rounds] for t, v in by.items()
                if all(k in v and not v[k]["errored"] for k in rounds)}
    tasks = [[(int(a["resolved"]), a["llm_calls"] or 0, a["seconds"]) for a in v] for v in complete.values()]
    rng = random.Random(SEED)
    point = stats(tasks)
    boots = [stats([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(1000)]
    ci = {k: pct([b[k] for b in boots if b[k] == b[k]]) for k in point if k != "failure_anchored_pairs"}
    dist = {str(k): sum(sum(a[0] for a in t) == k for t in tasks) for k in range(len(rounds) + 1)}

    def restart_given(flag):
        num = den = 0
        for v in complete.values():
            for i, a in enumerate(v):
                if a["resolved"] or not flag(a):
                    continue
                for j, b in enumerate(v):
                    if i != j:
                        den += 1; num += b["resolved"]
        return {"pairs": den, "q": round(num / den, 4) if den else None}

    flagged = {"loop_flag_by_20_calls": restart_given(lambda a: a["loop_flag_by_20_calls"]),
               "no_loop_flag": restart_given(lambda a: not a["loop_flag_by_20_calls"]),
               "timed_out": restart_given(lambda a: a["timed_out"]),
               "not_timed_out": restart_given(lambda a: not a["timed_out"]),
               "context_window": restart_given(lambda a: a.get("failure_kind") == "context_window"),
               "test_patch_conflict": restart_given(lambda a: a.get("failure_kind") == "test_patch_conflict"),
               "clean_failure_no_error": restart_given(lambda a: a.get("failure_kind") == "none"),
               "gate_stop": restart_given(lambda a: a["gate_stop"]),
               "no_gate_stop": restart_given(lambda a: not a["gate_stop"])}
    def boot_q(flag, reps=2000):
        vals = list(complete.values())
        rng2 = random.Random(SEED + 7)
        out = []
        for _ in range(reps):
            num = den = 0
            for v in (vals[rng2.randrange(len(vals))] for _ in vals):
                for i, a in enumerate(v):
                    if a["resolved"] or not flag(a):
                        continue
                    for j, b in enumerate(v):
                        if i != j:
                            den += 1; num += b["resolved"]
            if den:
                out.append(num / den)
        return pct(out)

    for key, flag in (("loop_flag_by_20_calls", lambda a: a["loop_flag_by_20_calls"]),
                      ("no_loop_flag", lambda a: not a["loop_flag_by_20_calls"]),
                      ("timed_out", lambda a: a["timed_out"]),
                      ("not_timed_out", lambda a: not a["timed_out"])):
        flagged[key]["interval_95"] = boot_q(flag)
        flagged[key]["failed_attempts"] = sum(flag(a) and not a["resolved"] for v in complete.values() for a in v)
        flagged[key]["tasks"] = sum(any(flag(a) and not a["resolved"] for a in v) for v in complete.values())
    repos = {}
    for repo in sorted({a["repo"] for v in complete.values() for a in v}):
        sub = [[(int(a["resolved"]), a["llm_calls"] or 0, a["seconds"]) for a in v] for v in complete.values() if v[0]["repo"] == repo]
        repos[repo] = {"tasks": len(sub), **{k: round(v, 4) for k, v in stats(sub).items()}} if len(sub) > 1 else {"tasks": len(sub)}
    result = {
        "run": name, "attempts": len(rows), "rounds": rounds, "tasks_complete": len(tasks),
        "tasks_excluded_for_errors": len(by) - len(complete),
        "solved_k_of_n": dist, "point": {k: round(v, 4) for k, v in point.items()}, "interval_95": ci,
        "restart_q_by_failure_signature": flagged, "by_repo": repos,
        "frontier_wall_seconds": frontier(tasks, [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], 1000, random.Random(SEED + 1)),
        "targeted_restart_wall_seconds": targeted(complete, [0.4, 0.5, 0.6, 0.7, 0.8], 1000, random.Random(SEED + 2)),
        "mean_seconds_success": round(sum(a[2] for t in tasks for a in t if a[0]) / max(1, sum(a[0] for t in tasks for a in t)), 1),
        "mean_seconds_failure": round(sum(a[2] for t in tasks for a in t if not a[0]) / max(1, sum(1 - a[0] for t in tasks for a in t)), 1),
        "loop_share_of_failures": round(sum(a["loop_flag_by_20_calls"] and not a["resolved"] for v in complete.values() for a in v)
                                        / max(1, sum(not a["resolved"] for v in complete.values() for a in v)), 4),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gemma_rollouts_{name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_repo",)}, indent=1))


if __name__ == "__main__":
    if ARGV[1] == "derive":
        derive(Path(ARGV[2]), ARGV[3])
    else:
        analyze(ARGV[2])
