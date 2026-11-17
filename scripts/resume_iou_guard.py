"""Resume the identical WMX trajectories with a cache that fits all six matches."""
import fcntl
from pathlib import Path

from experiments.campaign_protocol import publish
from experiments.gpu_iou import status, train_parent, train_refinement
from experiments.gpu_runtime import accelerated_preprocessing, configure_device
from experiments.iou_followup_training import OUTPUT, freeze
from experiments.robustness import _digest
from possession.storage import sha256_file
from possession.training_session import HalfCache


def main():
    plan = freeze()
    config = plan["arms"]["guard"]
    output = OUTPUT / "guard"
    publish(OUTPUT / "guard-cache-amendment.json", {
        "script_sha256": sha256_file(Path(__file__)), "protocol_sha256": _digest(plan),
        "original_cache_bytes": 1_300_000_000, "effective_guard_cache_bytes": 1_600_000_000,
        "measured_six_match_array_bytes": 1_491_499_841,
        "reason": "Fit all six training matches in the cache and release idle fold workers. "
                  "Resume authenticated model, optimizer, RNG and transcript state through the existing trainer.",
        "model_or_analysis_parameters_changed": False})
    configure_device("cuda")
    with (OUTPUT / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fine = HalfCache(Path(config["data_dir"]), 1_600_000_000)
        coarse = HalfCache(Path(config["coarse_data_dir"]), 200_000_000)
        fits = []
        with accelerated_preprocessing():
            for seed in config["seeds"]:
                directory, fit = train_parent(config, output, 0, seed, "coarse_player_only",
                                               config["coarse_spec"], coarse, coarse=True)
                fits.append({"directory": str(directory), **fit})
                for member in config["selection"]["members"]:
                    directory, fit = train_parent(config, output, 0, seed, member["id"], member["spec"], fine)
                    fits.append({"directory": str(directory), **fit})
                    for variant in config["variants"]:
                        child_dir, child = train_refinement(config, output, directory, fit, variant,
                                                            member["spec"], fine)
                        fits.append({"directory": str(child_dir), **child})
        if len(fits) != 14:
            raise ValueError("incomplete WMX grid")
        # Keep the original fold-manifest ordering; the aggregate is sorted.
        publish(output / "training-fold0.json", {"protocol_sha256": _digest(config), "fits": fits})
        fits.sort(key=lambda r: (r["identity"]["fold"], r["identity"]["base_seed"],
                                 r["identity"]["member"], r["identity"]["role"]))
        publish(output / "training.json", {"protocol_sha256": _digest(config), "fits": fits})
        status(output, "training_complete", fold=0, completed_fits=14)
        status(OUTPUT, "training_complete", completed_fits=80)


if __name__ == "__main__":
    main()
