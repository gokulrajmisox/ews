"""Feature-group and feature-count ablation experiments.

The experiment compares clinically meaningful feature families without touching
checked-in production model artifacts. It requires cached or supplied raw data.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.research_common import ResearchDataUnavailable, json_dump, load_train_validation, skipped_result


def _fit_score(x_train, y_train, x_val, y_val, feature_cols, config):
    import xgboost as xgb

    params = dict(config["model"].get("xgboost_params", {}))
    params.setdefault("eval_metric", "logloss")
    params["n_jobs"] = 1
    model = xgb.XGBClassifier(**params)
    model.fit(x_train[feature_cols], y_train)
    probs = model.predict_proba(x_val[feature_cols])[:, 1]
    return {
        "feature_count": len(feature_cols),
        "roc_auc": round(float(roc_auc_score(y_val, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_val, probs)), 4),
        "features": feature_cols,
    }, model


def run_feature_ablation(
    config_path: str = "configs/config.yaml",
    output_path: str = "results/research/feature_ablation.json",
) -> dict[str, Any]:
    try:
        train_df, val_df, feature_cols = load_train_validation(config_path)
    except ResearchDataUnavailable as exc:
        return json_dump(
            output_path,
            skipped_result(
                "feature_and_ablation_experiments",
                str(exc),
                ["data/raw/set-a/*.txt", "data/raw/Outcomes-train.txt"],
            ),
        )

    from ml.preprocessing import load_config

    config = load_config(config_path)
    y_train = train_df["target"].astype(int).to_numpy()
    y_val = val_df["target"].astype(int).to_numpy()
    results = []

    def add_variant(variant_id: str, columns: list[str]) -> None:
        metrics, _ = _fit_score(train_df, y_train, val_df, y_val, columns, config)
        results.append({"variant_id": variant_id, **metrics})

    add_variant("full", feature_cols)
    add_variant("without_gcs", [c for c in feature_cols if not c.startswith("GCS_")])
    add_variant("without_icu_type", [c for c in feature_cols if c != "icu_type"])
    static = {"age", "gender", "weight", "icu_type"}
    add_variant("vitals_and_trajectory_only", [c for c in feature_cols if c not in static])
    vitals_only = [c for c in feature_cols if any(c.startswith(f"{v}_") for v in ["HR", "SysABP", "DiasABP", "MAP", "RespRate", "Temp", "GCS", "Urine", "SaO2", "Glucose"])]
    add_variant("vitals_only", vitals_only)

    full_metrics, full_model = _fit_score(train_df, y_train, val_df, y_val, feature_cols, config)
    importances = np.asarray(full_model.feature_importances_)
    ranked_features = [feature_cols[i] for i in np.argsort(importances)[::-1]]
    for k in [30, 50, 75, 100, len(feature_cols)]:
        columns = ranked_features[: min(k, len(ranked_features))]
        add_variant(f"top_{len(columns)}_features", columns)

    payload = {
        "status": "completed",
        "experiment": "feature_and_ablation_experiments",
        "baseline_preserved": True,
        "objective": "Compare PR-AUC and ROC-AUC under feature-family and top-k variants.",
        "results": results,
        "disclaimer": "Validation-only feature study. Results do not establish clinical utility or justify removing clinically meaningful variables without review.",
    }
    return json_dump(output_path, payload)


if __name__ == "__main__":
    print(json.dumps(run_feature_ablation(), indent=2))
