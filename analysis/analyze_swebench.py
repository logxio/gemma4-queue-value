import json
import posixpath
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "verified"
if SPLIT == "lite":
    ROOT = ROOT / "lite"
elif SPLIT != "verified":
    raise ValueError("split must be verified or lite")
ROWS = json.loads((ROOT / "audit_manifest.json").read_text())
SYSTEMS = ["sweagent", "agentless", "openhands"] if SPLIT == "verified" else ["sweagent", "openhands"]
RESULTS = {name: json.loads((ROOT / "results" / f"{name}.json").read_text()) for name in SYSTEMS}


def clean(path):
    path = path.strip(" `\"'()[],:;\n\r")
    path = re.sub(r"^/workspace/[^/]+/", "", path)
    path = re.sub(r"^/[^/]+__[^/]+/", "", path)
    path = re.sub(r"^/(?:workspace|testbed|repo)/", "", path)
    path = re.sub(r"^(?:testbed|repo)/", "", path)
    if path.startswith("./"):
        path = path[2:]
    if not re.search(r"\.[A-Za-z][A-Za-z0-9]*$", path):
        return None
    if ".." in path or "site-packages" in path:
        return None
    return path.lstrip("/")


def unique(paths):
    out = []
    for path in paths:
        path = clean(path)
        if path and path not in out:
            out.append(path)
    return out[:5]


def auxiliary(path):
    path = clean(path)
    if not path:
        return True
    base = posixpath.basename(path).lower()
    parts = path.lower().split("/")
    return (
        base.startswith(("reproduce", "repro_", "test_"))
        or base in {"repro.py", "test.py"}
        or "tests" in parts
        or "test" in parts
    )


def sweagent(path):
    rows = json.loads(path.read_text())["trajectory"]
    candidates = []
    created = set()
    for row in rows:
        action = row.get("action", "").strip()
        state = json.loads(row.get("state", "{}"))
        current = clean(state.get("open_file", ""))
        workdir = state.get("working_dir", "")
        def in_workdir(value):
            return posixpath.normpath(posixpath.join(workdir, value)) if not value.startswith("/") else value
        if action.startswith("create "):
            new_file = clean(in_workdir(action.splitlines()[0][7:]))
            if new_file:
                created.add(new_file)
                if "/" in new_file and not auxiliary(new_file):
                    break
        if action.startswith("edit ") and current and current not in created and not auxiliary(current):
            break
        if action.startswith("open "):
            candidates.append(in_workdir(action.splitlines()[0][5:]))
        if action.startswith("search_dir "):
            candidates.extend(re.findall(r"(?m)^(/[^\s()]+\.[A-Za-z0-9]+) \(\d+ matches?\)", row.get("observation", "")))
        if action.startswith("search_file "):
            parts = action.splitlines()[0].split()
            if len(parts) >= 3:
                candidates.append(in_workdir(parts[-1]))
    return unique(candidates)


def agentless(path):
    text = path.read_text(errors="replace")
    section = text.split("### Model Predicted Suspicious Files", 1)[-1].split("### Model Predicted Irrelevant Folders", 1)[0]
    api = re.search(r"(?m)^\d{4}-\d\d-\d\d [^\n]+ - INFO - API response ChatCompletion", section)
    if not api:
        raise ValueError("no localization API response")
    headers = list(re.finditer(r"(?m)^\d{4}-\d\d-\d\d [^\n]+ - INFO - ", section[api.end():]))
    if not headers:
        raise ValueError("no rendered localization response")
    tail = section[api.end():]
    answer = tail[headers[0].end():headers[1].start()] if len(headers) > 1 else tail[headers[0].end():]
    response = re.search(r"```[^\n]*\n(.*?)\n```", answer, re.S)
    if not response:
        raise ValueError("no fenced file list in localization response")
    return unique(response.group(1).splitlines())


def openhands(path):
    messages = json.loads(path.read_text())
    candidates = []
    created = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []):
            fn = call.get("function", {})
            try:
                arg = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                continue
            if fn.get("name") == "str_replace_editor":
                cmd = arg.get("command", "")
                candidate = clean(arg.get("path", ""))
                if cmd == "create" and candidate:
                    created.add(candidate)
                    if "/" in candidate and not auxiliary(candidate):
                        return unique(candidates)
                if cmd in {"str_replace", "insert"} and candidate and candidate not in created and not auxiliary(candidate):
                    return unique(candidates)
                if cmd == "view":
                    candidates.append(arg.get("path", ""))
            if fn.get("name") == "execute_bash":
                cmd = arg.get("command", "")
                if re.search(r"\b(?:sed|cat|head|tail|rg|grep)\b", cmd):
                    candidates.extend(re.findall(r"(?:/workspace/[^/]+/)?(?:[\w.-]+/)+[\w.-]+\.(?:py|js|ts|go|rs|java|c|cpp|h)", cmd))
                if re.search(r"\b(?:apply_patch|git apply)\b", cmd):
                    return unique(candidates)
    return unique(candidates)


PARSERS = {"sweagent": (sweagent, ".traj"), "agentless": (agentless, ".log"), "openhands": (openhands, ".json")}
PARSERS = {k: PARSERS[k] for k in SYSTEMS}
records = []
for row in ROWS:
    issue = row["instance_id"]
    gold = set(row["gold_files"])
    for system, (parser, ext) in PARSERS.items():
        path = ROOT / "trajs" / system / f"{issue}{ext}"
        if not path.exists():
            continue
        try:
            candidates = parser(path)
        except Exception as exc:
            print("PARSE_ERROR", system, issue, type(exc).__name__, str(exc)[:80])
            continue
        predicted = set(candidates)
        f1 = 2 * len(gold & predicted) / (len(gold) + len(predicted)) if predicted else 0
        hit1 = int(bool(candidates) and candidates[0] in gold)
        hit5 = int(bool(gold & predicted))
        outcome = int(issue in RESULTS[system]["resolved"])
        records.append(dict(instance_id=issue, repo=row["repo"], system=system,
                            gold=sorted(gold), candidates=candidates, f1=f1,
                            hit1=hit1, hit5=hit5, resolved=outcome))

(ROOT / "audit_rows.json").write_text(json.dumps(records, indent=2) + "\n")
print("records", len(records), "issues", len(set(r["instance_id"] for r in records)))
for system in PARSERS:
    part = [r for r in records if r["system"] == system]
    print(system, "n", len(part), "with_candidate", sum(bool(r["candidates"]) for r in part),
          "hit1", sum(r["hit1"] for r in part), "hit5", sum(r["hit5"] for r in part),
          "resolved", sum(r["resolved"] for r in part), "mean_f1", round(sum(r["f1"] for r in part) / len(part), 3) if part else 0)
    for metric in ["hit1", "hit5"]:
        high = [r for r in part if r[metric]]
        low = [r for r in part if not r[metric]]
        print("  ", metric, "resolved_if_hit", f"{sum(r['resolved'] for r in high)}/{len(high)}",
              "resolved_if_miss", f"{sum(r['resolved'] for r in low)}/{len(low)}")
