"""Action-prefix features from agent-visible actions; never opens a reference patch."""
import json
import posixpath
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
CHECKPOINT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
if CHECKPOINT not in (5, 10, 15):
    raise ValueError("checkpoint must be 5, 10, or 15")
PATH_RE = re.compile(r"(?<![\w])(?:/testbed/|/workspace/[^/]+/|/[\w.-]+__/)?((?:[\w.@+-]+/)+[\w.@+-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|cs|rb|php|kt|scala|vue|sh|xml|yaml|yml|json|md))\b")
TEST_RE = re.compile(r"(?i)(?:^|/)(?:test|tests|spec|specs|node_modules)/|(?:^|/)(?:test_|spec_|repro|reproduce)")
TEST_CMD = re.compile(r"(?i)\b(?:pytest|unittest|go test|cargo test|npm test|pnpm test|yarn test|phpunit|rspec|mvn test|gradle test)\b")
EDIT_CMD = re.compile(r"(?i)__EDIT__|\b(?:apply_patch|git apply|sed -i|perl -pi)\b|\bcat\s*>|\btee\s+")


def files(text):
    out = []
    for match in PATH_RE.finditer(text):
        path = posixpath.normpath(match.group(1)).lstrip("/")
        if ".." not in path.split("/") and not path.startswith(("site-packages/", "node_modules/")):
            out.append(path)
    return out


def features(actions, full_calls, issue, system, split, resolved):
    first = actions[:CHECKPOINT]
    paths = [path for action in first for path in files(action)]
    by_turn = [set(files(action)) for action in first]
    counts = Counter(paths)
    source = [path for path in paths if not TEST_RE.search(path)]
    revisited = len(set.intersection(*[s for s in by_turn if s])) if sum(bool(s) for s in by_turn) > 1 else 0
    seen_before_edit = set()
    edit_agreement = 0
    has_source_edit = 0
    for action in first:
        marker = re.search(r"__EDIT_PATH__:(\S+)", action)
        if marker:
            edited = files(marker.group(1))
            if edited and not TEST_RE.search(edited[0]):
                has_source_edit = 1
                edit_agreement = int(edited[0] in seen_before_edit)
                break
        seen_before_edit.update(files(action))
    return {
        "instance_id": issue, "repo": issue.rsplit("-", 1)[0], "system": system, "split": split,
        "resolved": int(resolved), "full": int(full_calls), "prefix": min(CHECKPOINT, int(full_calls)),
        "path_mentions": len(paths), "unique_paths": len(counts),
        "source_paths": len(set(source)), "test_path_share": sum(bool(TEST_RE.search(p)) for p in paths) / len(paths) if paths else 0,
        "focus": max(counts.values()) / len(paths) if paths else 0,
        "repeat_share": (len(paths) - len(counts)) / len(paths) if paths else 0,
        "revisited": revisited,
        "edit_agreement": edit_agreement, "has_source_edit": has_source_edit,
        "test_early": int(any(TEST_CMD.search(action) for action in first)),
        "edit_early": int(any(EDIT_CMD.search(action) or re.match(r"^(?:edit|create)\s", action) for action in first)),
    }


def sweagent(path):
    trace = json.loads(path.read_text())
    actions = []
    for turn in trace["trajectory"]:
        action = turn.get("action", "")
        if action.startswith("edit "):
            state = json.loads(turn.get("state", "{}"))
            edited = state.get("open_file", "")
            if edited:
                action += "\n__EDIT_PATH__:" + edited
        elif action.startswith("create "):
            action += "\n__EDIT_PATH__:" + action.splitlines()[0][7:]
        actions.append(action)
    return actions, trace["info"]["model_stats"]["api_calls"]


def agentless(path):
    raw = path.read_text(errors="replace")
    lines = re.findall(r"(?m)^.*? - INFO - API response ChatCompletion.*$", raw)
    return lines, len(lines)


def tool_actions(messages):
    output = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        chunks = []
        for call in msg.get("tool_calls", []):
            fn = call.get("function", {})
            try:
                arg = json.loads(fn.get("arguments", "{}"))
            except (ValueError, TypeError):
                continue
            chunks.append(str(arg.get("path", "")))
            chunks.append(str(arg.get("command", "")))
            if fn.get("name") == "str_replace_editor" and arg.get("command") in ("create", "str_replace", "insert"):
                chunks.append("__EDIT__")
                chunks.append("__EDIT_PATH__:" + str(arg.get("path", "")))
            elif fn.get("name") == "bash" and EDIT_CMD.search(str(arg.get("command", ""))):
                paths = files(str(arg.get("command", "")))
                if paths:
                    chunks.append("__EDIT_PATH__:" + paths[0])
        output.append("\n".join(chunks))
    return output


def openhands(path):
    actions = tool_actions(json.loads(path.read_text()))
    return actions, len(actions)


def multilingual(path, details):
    trace = json.loads(path.read_text())
    return tool_actions(trace["messages"]), details["api_calls"]


def build():
    rows = []
    verified = None
    for split in ("verified", "lite"):
        base = ROOT if split == "verified" else ROOT / "lite"
        systems = ("sweagent", "agentless", "openhands", "mini") if split == "verified" else ("sweagent", "openhands")
        for system in systems:
            if system == "mini":
                entry = "20260217_mini-v2.0.0_claude-4-5-haiku-high"
                details = json.loads((ROOT / "experiments" / "evaluation" / "verified" / entry / "per_instance_details.json").read_text())
                for path in sorted((base / "trajs" / "mini").glob("*.traj.json")):
                    issue = path.name.removesuffix(".traj.json")
                    actions, full = multilingual(path, details[issue])
                    rows.append(features(actions, full, issue, system, split, details[issue]["resolved"]))
                continue
            resolved = set(json.loads((base / "results" / f"{system}.json").read_text())["resolved"])
            suffix = {"sweagent": ".traj", "agentless": ".log", "openhands": ".json"}[system]
            parser = {"sweagent": sweagent, "agentless": agentless, "openhands": openhands}[system]
            for path in sorted((base / "trajs" / system).glob("*" + suffix)):
                issue = path.name.removesuffix(suffix)
                if split == "lite" and issue in verified:
                    continue
                actions, full = parser(path)
                rows.append(features(actions, full, issue, system, split, issue in resolved))
        if split == "verified":
            verified = {r["instance_id"] for r in rows}
    details = json.loads((ROOT / "multilingual_per_instance_details.json").read_text())
    for issue, detail in sorted(details.items()):
        path = ROOT / "multilingual" / "trajs" / f"{issue}.traj.json"
        actions, full = multilingual(path, detail)
        rows.append(features(actions, full, issue, "mini", "multilingual", detail["resolved"]))
    (ROOT / f"runtime_rows_{CHECKPOINT}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"checkpoint": CHECKPOINT, "rows": len(rows), "groups": {f"{split}/{system}": sum(r["split"] == split and r["system"] == system for r in rows) for split, system in sorted({(r["split"], r["system"]) for r in rows})}}, indent=2))


if __name__ == "__main__":
    build()
