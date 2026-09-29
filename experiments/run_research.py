"""Run all requested SilentWindow research experiments in a safe order."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.ensemble import run_ensemble
from experiments.feature_ablation import run_feature_ablation
from experiments.future_window_labels import run_future_window_label_experiment
from experiments.tune_xgboost import run_tuning
from ml.policy_sweep import run_and_save


def run_all(output_path: str = "results/research/experiment_manifest.json") -> dict:
    results = {
        "policy_sweep": run_and_save(output_path="results/research/policy_sweep_expanded.json"),
        "class_weight_xgboost_tuning": run_tuning(),
        "feature_ablation": run_feature_ablation(),
        "future_window_labels": run_future_window_label_experiment(),
        "ensemble": run_ensemble(),
    }
    manifest = {
        "status": "completed",
        "baseline_preserved": True,
        "experiments": {
            name: {
                "status": payload.get("status"),
                "experiment": payload.get("experiment", name),
                "result_path": {
                    "policy_sweep": "results/research/policy_sweep_expanded.json",
                    "class_weight_xgboost_tuning": "results/research/xgboost_tuning.json",
                    "feature_ablation": "results/research/feature_ablation.json",
                    "future_window_labels": "results/research/future_window_labels.json",
                    "ensemble": "results/research/ensemble_benchmark.json",
                }[name],
            }
            for name, payload in results.items()
        },
        "results": results,
        "disclaimer": "Research outputs are retrospective or validation-only. No experiment automatically replaces models/ artifacts or runtime policy.",
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    manifest = run_all()
    print(json.dumps({name: value["status"] for name, value in manifest["experiments"].items()}, indent=2))
