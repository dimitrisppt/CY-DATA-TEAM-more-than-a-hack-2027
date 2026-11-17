"""Read the detached tuning campaign without touching its GPU worker."""
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.campaign_protocol import read_record


def main():
    output = Path(__file__).resolve().parents[1] / "artifacts/gpu-tuning-8h-v1"
    path = output / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
    clock = read_record(output / "clock.json")
    if clock:
        status["deadline_utc"] = datetime.fromtimestamp(clock["deadline_unix"], UTC).isoformat()
        status["hours_left"] = round(max(0., clock["deadline_unix"] - time.time()) / 3600, 3)
    status["screened_recipes"] = len(list((output / "screen").glob("trial_*/result.json")))
    status["full_budget_recipes"] = len(list((output / "full").glob("trial_*/result.json")))
    status["decoder_trials"] = len(list((output / "polish").glob("*/trial_*.json")))
    for name in ("selection", "confirmation", "guard", "exit", "error"):
        status[f"{name}_saved"] = (output / f"{name}.json").exists()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
