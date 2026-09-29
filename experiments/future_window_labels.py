"""Future-window target redesign using the available endpoint as an explicit proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.research_common import ResearchDataUnavailable, json_dump, require_raw_data, skipped_result
from ml.preprocessing import load_config, load_patient_raw
from ml.labeling import OutcomeLabeler


def run_future_window_label_experiment(
    config_path: str = "configs/config.yaml",
    output_path: str = "results/research/future_window_labels.json",
    horizons: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0),
) -> dict[str, Any]:
    config = load_config(config_path)
    try:
        require_raw_data(config)
    except ResearchDataUnavailable as exc:
        return json_dump(
            output_path,
            skipped_result(
                "future_window_proxy_labels",
                str(exc),
                ["data/raw/set-a/*.txt", "data/raw/Outcomes-train.txt"],
            ),
        )

    labeler = OutcomeLabeler(config["data"]["outcomes_file"])
    records_dir = Path(config["data"]["records_dir"])
    summary = []
    for horizon in horizons:
        rows = 0
        positives = 0
        patients = 0
        positive_patients = 0
        for path in sorted(records_dir.glob("*.txt")):
            try:
                patient_id = int(path.stem)
                _, observations = load_patient_raw(str(path))
            except (ValueError, OSError):
                continue
            if observations.empty or "time_hours" not in observations:
                continue
            patient_end = float(observations["time_hours"].max())
            outcome = labeler.get_patient_outcome(patient_id)
            patients += 1
            if outcome["in_hospital_death"] == 1:
                positive_patients += 1
            # Build an hourly evaluation grid using the same endpoint proxy.
            import pandas as pd
            features = pd.DataFrame({"eval_time": [float(t) for t in range(2, int(patient_end) + 1)]})
            labeled = labeler.assign_future_window_proxy_labels(features, patient_id, patient_end, horizon)
            rows += len(labeled)
            positives += int(labeled["future_window_proxy_target"].sum())
        summary.append({
            "horizon_hours": float(horizon),
            "patients": patients,
            "positive_outcome_patients": positive_patients,
            "rows": rows,
            "proxy_positive_rows": positives,
            "proxy_positive_rate": round(positives / rows, 4) if rows else 0.0,
        })

    payload = {
        "status": "completed",
        "experiment": "future_window_proxy_labels",
        "baseline_preserved": True,
        "horizons_hours": list(horizons),
        "results": summary,
        "definition": "Positive means an in-hospital death outcome exists and the available telemetry endpoint proxy falls within the future horizon.",
        "critical_limitation": "The dataset has no exact deterioration/death timestamp; these are proxy labels and must not be presented as validated early-warning targets.",
    }
    return json_dump(output_path, payload)


if __name__ == "__main__":
    print(json.dumps(run_future_window_label_experiment(), indent=2))
