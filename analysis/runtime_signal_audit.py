"""Cross-validated prediction of resolution from single runtime-prefix signals."""
import json

from runtime_replay import ROWS, auc, bucket, fit, predict

SIGNALS = ("path_mentions", "unique_paths", "source_paths", "focus", "repeat_share",
           "test_path_share", "revisited", "edit_early", "test_early",
           "edit_agreement", "has_source_edit")


def audit():
    output = {}
    for system in ("sweagent", "agentless", "openhands", "mini"):
        output[system] = {}
        for checkpoint in (5, 10, 15):
            train = [r for r in ROWS[checkpoint] if r["system"] == system
                     and r["split"] == "verified" and bucket(r["instance_id"]) < 153]
            result = {}
            for signal in SIGNALS:
                predictions = []
                for fold in range(3):
                    fitting = [r for r in train if bucket(r["instance_id"], "fold") % 3 != fold]
                    validation = [r for r in train if bucket(r["instance_id"], "fold") % 3 == fold]
                    model = fit(fitting, (signal,))
                    predictions += [(r["resolved"], float(p)) for r, p in
                                    zip(validation, predict(model, validation, (signal,)))]
                result[signal] = round(auc([y for y, _ in predictions], [p for _, p in predictions]), 3)
            output[system][str(checkpoint)] = result
    return output


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
