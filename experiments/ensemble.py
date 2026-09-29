"""Optional ensemble benchmark; production artifacts remain untouched."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.research_common import ResearchDataUnavailable, json_dump, load_train_validation, skipped_result


def _score(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
    }


def run_ensemble(
    config_path: str = "configs/config.yaml",
    output_path: str = "results/research/ensemble_benchmark.json",
) -> dict[str, Any]:
    try:
        train_df, val_df, feature_cols = load_train_validation(config_path)
    except ResearchDataUnavailable as exc:
        return json_dump(
            output_path,
            skipped_result(
                "optional_model_ensemble",
                str(exc),
                ["data/raw/set-a/*.txt", "data/raw/Outcomes-train.txt"],
            ),
        )

    import xgboost as xgb

    y_train = train_df["target"].astype(int).to_numpy()
    y_val = val_df["target"].astype(int).to_numpy()
    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    xgb_params = {
        **config_from_yaml(config_path)["model"].get("xgboost_params", {}),
        "eval_metric": "logloss",
        "n_jobs": 1,
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(x_train, y_train)
    xgb_probs = xgb_model.predict_proba(x_val)[:, 1]

    lr_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    lr_model.fit(x_train, y_train)
    lr_probs = lr_model.predict_proba(x_val)[:, 1]

    models = [_score("xgboost", y_val, xgb_probs), _score("logistic_regression", y_val, lr_probs)]
    probabilities = {"xgboost": xgb_probs, "logistic_regression": lr_probs}
    weights = {"xgboost": 0.6, "logistic_regression": 0.4}

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(x_train, y_train)
        probabilities["lightgbm"] = lgb_model.predict_proba(x_val)[:, 1]
        models.append(_score("lightgbm", y_val, probabilities["lightgbm"]))
        weights = {"xgboost": 0.5, "lightgbm": 0.3, "logistic_regression": 0.2}
    except ImportError:
        pass

    ensemble_probs = sum(weights[name] * probabilities[name] for name in weights)
    models.append(_score("weighted_ensemble", y_val, ensemble_probs))
    payload = {
        "status": "completed",
        "experiment": "optional_model_ensemble",
        "baseline_preserved": True,
        "weights": weights,
        "models": models,
        "disclaimer": "Validation-only benchmark. The ensemble is not promoted or saved as the production model.",
    }
    return json_dump(output_path, payload)


def config_from_yaml(config_path: str) -> dict:
    from ml.preprocessing import load_config
    return load_config(config_path)


if __name__ == "__main__":
    print(json.dumps(run_ensemble(), indent=2))
