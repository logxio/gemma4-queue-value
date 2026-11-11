"""Freeze issue IDs for the Gemma 4 prospective queue experiment.

The input is the official public tasks.jsonl. Only issue IDs and repo labels
enter the output; task text, reference patches, and tests do not.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(instance_id: str, salt: str) -> str:
    return hashlib.sha256((instance_id + salt).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_jsonl", type=Path)
    args = parser.parse_args()
    source = args.tasks_jsonl.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    issues = [{"instance_id": row["instance_id"], "repo": row["repo"]}
              for row in rows]
    assert len(issues) == 129 and len({x["instance_id"] for x in issues}) == 129
    assert sum(x["repo"] == "encode/httpx" for x in issues) == 1
    eligible = [x for x in issues if x["repo"] != "encode/httpx"]
    pilot = sorted(eligible,
                   key=lambda x: (digest(x["instance_id"], "paper-s2-pilot"),
                                  x["instance_id"]))[:12]
    pilot_ids = {x["instance_id"] for x in pilot}
    formal = sorted((x for x in issues if x["instance_id"] not in pilot_ids),
                    key=lambda x: (digest(x["instance_id"], "paper-s2-formal"),
                                   x["instance_id"]))
    assert len(formal) == 117 and any(x["repo"] == "encode/httpx" for x in formal)
    out = {
        "purpose": "frozen pilot and formal issue order before Gemma outcomes",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "pilot_salt": "paper-s2-pilot",
        "formal_salt": "paper-s2-formal",
        "pilot": pilot,
        "formal": formal,
    }
    dest = Path(__file__).with_name("gemma_issue_split.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": str(dest), "pilot": len(pilot),
                      "formal": len(formal), "source_sha256": out["source_sha256"]}))


if __name__ == "__main__":
    main()
