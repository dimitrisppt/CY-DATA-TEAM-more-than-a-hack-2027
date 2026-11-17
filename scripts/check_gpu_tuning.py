"""Exercise final assessment/report wiring with tiny models on the actual inputs.

This is a functional smoke test, not a performance experiment. It never supplies
search scores to the eight-hour campaign.
"""
import copy
from pathlib import Path

from experiments.campaign_protocol import publish
from experiments.gpu_runtime import accelerated_preprocessing, configure_device
from experiments.gpu_tuning import Campaign, generate_report
from experiments.gpu_tuning_space import pipeline_anchor
from experiments.robustness import _digest


def main():
    repo = Path(__file__).resolve().parents[1]
    output = repo / "artifacts/gpu-tuning-integration-v1"
    output.mkdir(exist_ok=True)
    configure_device()
    campaign = Campaign(repo, output, .5)
    spec = copy.deepcopy(campaign.plan["members"][0]["spec"])
    spec["model_kwargs"].update(width=8, dilations=[1, 2])
    spec["training_args"]["center_size"] = 64
    spec["schedule"].update(total_steps=200, warmup_steps=10)
    recipe = {"spec": spec, "spell_weight": .25, "short_sampling_probability": .25, "export": "ema"}
    proposal = {"id": "functional_smoke", "recipe": recipe, "smoke_test_only": True}
    selection = {"id": proposal["id"], "steps": 200,
                 "pipeline": pipeline_anchor(campaign.plan["decoder"], .5),
                 "proposal": proposal, "search_scores": {"objective": 0.},
                 "smoke_test_only": True, "protocol_sha256": _digest(campaign.plan)}
    publish(output / "selection.json", selection)
    with accelerated_preprocessing():
        campaign.assess("inner")
        campaign.assess("guard")
        verification = campaign.verify()
        publish(output / "verification.json", {"smoke_test_only": True, **verification})
        generate_report(output)
    print("End-to-end CUDA assessment, verification, and report smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
