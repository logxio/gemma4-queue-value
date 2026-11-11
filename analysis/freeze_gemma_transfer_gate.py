"""Freeze the SWE-agent runtime gate for a prospective Gemma 4 transfer test.

Uses only the original Verified training split. Gemma development outcomes never
enter this artifact. The issue queue experiment has a separate protocol.
"""
import hashlib
import json
from pathlib import Path

from runtime_replay import ROWS, bucket, eligible, fit, predict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".ai" / "swebench" / "runtime_rows_10.json"
OUT = ROOT / "analysis" / "gemma_transfer_gate.json"
CHECKPOINT = 10
FEATURES = ("source_paths",)


def main():
    training = [r for r in ROWS[CHECKPOINT]
                if r["system"] == "sweagent" and r["split"] == "verified"
                and bucket(r["instance_id"]) < 153]
    assert len(training) == 280, len(training)
    model = fit(training, FEATURES)
    scored = sorted(float(p) + bucket(r["instance_id"], "tie") * 1e-9
                    for r, p in zip(training, predict(model, training, FEATURES))
                    if eligible(r))
    threshold = scored[max(0, int(.2 * len(scored)) - 1)] + 1e-12
    mean, scale, weights, bias = model
    artifact = {
        "purpose": "prospective cross-agent transfer; no Gemma outcome calibration",
        "source": "SWE-agent Verified, 280 issue-ID-hash training issues",
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "checkpoint_actions": CHECKPOINT,
        "features": list(FEATURES),
        "transform": "log1p(source_paths); standardize by mean and scale; clip to [-5,5]; sigmoid",
        "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": weights.tolist(), "bias": float(bias),
        "threshold": float(threshold),
        "training_issue_count": len(training),
        "eligible_training_count": len(scored),
        "stop_target_fraction": .2,
        "decision": "stop when score plus hash tie-break is below threshold after 10 actions and task is unresolved",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "train": len(training),
                      "eligible": len(scored), "threshold": threshold}))


if __name__ == "__main__":
    main()
