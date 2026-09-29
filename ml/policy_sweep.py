"""Alert-policy operating-point analysis over chronological replay records.

The policy sweep is validation-only analysis. It never changes the checked-in
model or runtime configuration. Results are meaningful only for the supplied
retrospective replay artifact and must not be treated as clinical validation.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ml.evidence_accumulator import SequentialEvidenceAccumulator
from ml.preprocessing import load_config

REQUIRED_COLUMNS = {
    "patient_id",
    "time_hours",
    "calibrated_risk",
    "raw_risk",
    "mean_credibility",
    "actual_outcome",
}


def _validate_records(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Replay records are missing columns: {sorted(missing)}")


def _score_policy(df: pd.DataFrame, config: dict, policy: Mapping[str, float]) -> dict:
    policy_config = copy.deepcopy(config)
    accum_cfg = policy_config.setdefault("evidence_accumulator", {})
    accum_cfg.update(
        {
            "watch_threshold": float(policy["watch_threshold"]),
            "alert_threshold": float(policy["alert_threshold"]),
            "decay_rate": float(policy["decay_rate"]),
            "persistence_boost_factor": float(policy["persistence_boost_factor"]),
        }
    )

    patient_alerted: dict[int, bool] = {}
    patient_outcome: dict[int, int] = {}
    total_alerts = 0
    false_alert_events = 0
    for patient_id, patient_df in df.groupby("patient_id", sort=False):
        accumulator = SequentialEvidenceAccumulator(policy_config)
        patient_df = patient_df.sort_values("time_hours")
        for row in patient_df.itertuples(index=False):
            snapshot = accumulator.update(
                time_hours=float(row.time_hours),
                calibrated_risk=float(row.calibrated_risk),
                raw_risk=float(row.raw_risk),
                credibility_weight=float(row.mean_credibility),
            )
            if snapshot.alert_fired:
                total_alerts += 1
                patient_alerted[int(patient_id)] = True
                if int(row.actual_outcome) == 0:
                    false_alert_events += 1
        patient_outcome[int(patient_id)] = int(patient_df["actual_outcome"].iloc[-1])

    positives = sum(value == 1 for value in patient_outcome.values())
    detected_positive = sum(
        patient_alerted.get(pid, False) and outcome == 1
        for pid, outcome in patient_outcome.items()
    )
    alerted_patients = sum(patient_alerted.values())
    total_patient_days = max(
        1e-9, float(df.groupby("patient_id")["time_hours"].max().sum()) / 24.0
    )
    sensitivity = detected_positive / positives if positives else 0.0
    precision = detected_positive / alerted_patients if alerted_patients else 0.0
    return {
        **{key: round(float(value), 4) for key, value in policy.items()},
        "total_alerts": int(total_alerts),
        "false_alert_events": int(false_alert_events),
        "alerted_patients": int(alerted_patients),
        "positive_patients": int(positives),
        "positive_patients_detected": int(detected_positive),
        "sensitivity": round(float(sensitivity), 4),
        "precision": round(float(precision), 4),
        "alerts_per_patient_day": round(float(total_alerts / total_patient_days), 4),
    }


def pareto_frontier(results: Sequence[Mapping[str, float]]) -> list[dict]:
    """Return policies not dominated on sensitivity/precision/alert burden."""
    frontier = []
    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            no_worse = (
                other["sensitivity"] >= candidate["sensitivity"]
                and other["precision"] >= candidate["precision"]
                and other["false_alert_events"] <= candidate["false_alert_events"]
            )
            strictly_better = (
                other["sensitivity"] > candidate["sensitivity"]
                or other["precision"] > candidate["precision"]
                or other["false_alert_events"] < candidate["false_alert_events"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(dict(candidate))
    return sorted(
        frontier,
        key=lambda row: (
            -row["sensitivity"],
            -row["precision"],
            row["false_alert_events"],
        ),
    )


def sweep_alert_policies(
    records_path: str = "results/test_eval_records.parquet",
    watch_thresholds: Iterable[float] = (0.30, 0.35, 0.40, 0.45),
    alert_thresholds: Iterable[float] = (0.55, 0.60, 0.65, 0.70, 0.75),
    decay_rates: Iterable[float] = (0.80, 0.85, 0.88, 0.92, 0.95),
    persistence_boosts: Iterable[float] = (1.00, 1.10, 1.25, 1.40, 1.60),
    config_path: str = "configs/config.yaml",
    max_policies: int | None = None,
) -> dict:
    """Evaluate the Cartesian product of alert operating points."""
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"Replay records not found: {records_path}")
    df = pd.read_parquet(records_path)
    _validate_records(df)
    config = load_config(config_path)

    policies = [
        {
            "watch_threshold": float(watch),
            "alert_threshold": float(alert),
            "decay_rate": float(decay),
            "persistence_boost_factor": float(persistence),
        }
        for watch, alert, decay, persistence in itertools.product(
            watch_thresholds, alert_thresholds, decay_rates, persistence_boosts
        )
        if float(watch) < float(alert)
    ]
    if max_policies is not None:
        policies = policies[: int(max_policies)]

    results = []
    for index, policy in enumerate(policies, start=1):
        scored = _score_policy(df, config, policy)
        scored["policy_id"] = f"policy_{index:04d}"
        results.append(scored)

    frontier = pareto_frontier(results)
    current = config["evidence_accumulator"]
    current_policy = _score_policy(
        df,
        config,
        {
            "watch_threshold": current["watch_threshold"],
            "alert_threshold": current["alert_threshold"],
            "decay_rate": current["decay_rate"],
            "persistence_boost_factor": current["persistence_boost_factor"],
        },
    )
    current_policy["policy_id"] = "current_config"
    return {
        "status": "completed",
        "disclaimer": "Retrospective operating-point analysis only; not clinical validation or automatic threshold selection.",
        "records_path": records_path,
        "evaluated_policies": len(results),
        "grid": {
            "watch_thresholds": [float(x) for x in watch_thresholds],
            "alert_thresholds": [float(x) for x in alert_thresholds],
            "decay_rates": [float(x) for x in decay_rates],
            "persistence_boost_factors": [float(x) for x in persistence_boosts],
        },
        "current_policy": current_policy,
        "pareto_frontier": frontier,
        "results": results,
    }


def sweep_alert_thresholds(
    records_path: str = "results/test_eval_records.parquet",
    thresholds: Iterable[float] = (0.55, 0.65, 0.75, 0.85, 0.95),
    config_path: str = "configs/config.yaml",
) -> list[dict]:
    """Backward-compatible alert-only sweep used by the API."""
    expanded = sweep_alert_policies(
        records_path=records_path,
        watch_thresholds=(load_config(config_path)["evidence_accumulator"]["watch_threshold"] ,),
        alert_thresholds=thresholds,
        decay_rates=(load_config(config_path)["evidence_accumulator"]["decay_rate"],),
        persistence_boosts=(load_config(config_path)["evidence_accumulator"]["persistence_boost_factor"],),
        config_path=config_path,
    )
    return expanded["results"]


def run_and_save(output_path: str = "results/policy_sweep_expanded.json", **kwargs) -> dict:
    result = sweep_alert_policies(**kwargs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    result = run_and_save()
    print(json.dumps({
        "status": result["status"],
        "evaluated_policies": result["evaluated_policies"],
        "pareto_frontier_size": len(result["pareto_frontier"]),
    }, indent=2))
