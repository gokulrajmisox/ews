"""Class-weight and XGBoost tuning experiment.

This module writes research results only. It never overwrites models/ artifacts.
It requires the separately supplied raw dataset or cached train/validation
features, which are intentionally absent from the repository checkout.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from experiments.research_common import (
    ResearchDataUnavailable,
    json_dump,
    load_train_validation,
    skipped_result,
)


def _metric_row(name: str, params: dict[str, Any], y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    threshold = 0.5
    predictions = (probs >= threshold).astype(int)
    return {
        "trial": name,
        "params": params,
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "recall_at_0.5": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "precision_at_0.5": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "positive_rate_at_0.5": round(float(predictions.mean()), 4),
    }


def run_tuning(
    config_path: str = "configs/config.yaml",
    output_path: str = "results/research/xgboost_tuning.json",
    max_trials: int = 24,
) -> dict[str, Any]:
    try:
        train_df, val_df, feature_cols = load_train_validation(config_path)
    except ResearchDataUnavailable as exc:
        return json_dump(
            output_path,
            skipped_result(
                "class_weight_and_xgboost_tuning",
                str(exc),
                ["data/raw/set-a/*.txt", "data/raw/Outcomes-train.txt"],
            ),
        )

    import xgboost as xgb

    x_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int).to_numpy()
    x_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int).to_numpy()
    empirical_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "n_estimators": [200, 400, 600],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "gamma": [0.0, 0.1, 0.3],
        "scale_pos_weight": [3.0, 5.0, 6.0, round(empirical_weight, 4)],
    }
    candidates = list(itertools.islice(itertools.product(*grid.values()), max_trials))
    names = list(grid)
    results = []
    for index, values in enumerate(candidates, start=1):
        params = dict(zip(names, values))
        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_val)[:, 1]
        results.append(_metric_row(f"trial_{index:03d}", params, y_val, probs))

    ranked = sorted(results, key=lambda row: (row["pr_auc"], row["recall_at_0.5"]), reverse=True)
    payload = {
        "status": "completed",
        "experiment": "class_weight_and_xgboost_tuning",
        "baseline_preserved": True,
        "feature_count": len(feature_cols),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "empirical_scale_pos_weight": round(empirical_weight, 4),
        "objective": "PR-AUC first, then recall at threshold 0.5; accuracy is intentionally not optimized.",
        "trials_evaluated": len(results),
        "best_trial": ranked[0] if ranked else None,
        "trials": ranked,
        "disclaimer": "Validation-only tuning. Do not promote a trial to production without locked temporal test evaluation and calibration review.",
    }
    return json_dump(output_path, payload)


if __name__ == "__main__":
    print(json.dumps(run_tuning(), indent=2))
