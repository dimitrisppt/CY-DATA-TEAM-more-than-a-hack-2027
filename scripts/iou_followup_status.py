"""Read-only progress for the 80-fit CUDA follow-up."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts/gpu-iou-followup-v1"


def progress():
    plan = json.loads((ROOT / "protocol.json").read_text())["payload"]
    path = ROOT / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    finished = state.get("status") == "complete"
    endpoint = state["updated_unix"] if finished else time.time()
    result = {"status": "complete" if finished else "running",
              "elapsed_minutes": (endpoint - plan["frozen_unix"]) / 60, "arms": {}}
    for arm, expected in plan["expected_new_fits"].items():
        directory = ROOT / arm
        states = {}
        for path in (directory / "fits").rglob("state_*.pt"):
            if path.with_suffix(".pt.sha256.json").exists():
                states[path.parent] = max(states.get(path.parent, 0), int(path.stem.split("_")[-1]))
        updates = sum(step if path.relative_to(directory / "fits").parts[0] in ("coarse", "parent")
                      else step - 7200 for path, step in states.items())
        fold_states = [json.loads(p.read_text()) for p in sorted(directory.glob("status-fold*.json"))]
        result["arms"][arm] = {
            "complete_fits": len(list((directory / "fits").rglob("fit.json"))), "expected_fits": expected,
            "completed_updates": updates,
            "cached_predictions": len(list((directory / "inference").rglob("*.npz"))),
            "scored": (directory / "result.json").exists(),
            "active": [] if finished else [s for s in fold_states if s.get("stage") != "training_complete"],
        }
    result["complete_fits"] = sum(r["complete_fits"] for r in result["arms"].values())
    result["completed_updates"] = sum(r["completed_updates"] for r in result["arms"].values())
    result["expected_updates"] = plan["expected_new_updates"]
    result["update_fraction"] = result["completed_updates"] / result["expected_updates"]
    return result


if __name__ == "__main__":
    print(json.dumps(progress(), indent=2))
