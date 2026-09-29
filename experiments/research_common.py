"""Shared helpers for non-destructive SilentWindow research experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.preprocessing import get_patient_splits, load_config, load_patient_raw
from ml.train import build_split_dataset
from ml.trust_layer import TrustAwareSignalProcessor
from ml.features import FeatureExtractor
from ml.labeling import OutcomeLabeler

META_COLUMNS = {
    "patient_id",
    "eval_time",
    "target",
    "sofa",
    "saps",
    "length_of_stay",
    "survival",
}


class ResearchDataUnavailable(RuntimeError):
    """Raised when an experiment requires the excluded raw dataset."""


def require_raw_data(config: dict) -> None:
    records_dir = Path(config["data"]["records_dir"])
    outcomes_file = Path(config["data"]["outcomes_file"])
    if not records_dir.exists() or not outcomes_file.exists():
        raise ResearchDataUnavailable(
            "Raw PhysioNet records are not present. Place the permitted dataset at "
            f"{records_dir} and {outcomes_file} before running data-dependent experiments."
        )


def load_train_validation(config_path: str = "configs/config.yaml") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached features or build them from the separately supplied raw data."""
    config = load_config(config_path)
    processed_dir = Path(config["data"]["processed_dir"])
    train_cache = processed_dir / "train_features.parquet"
    val_cache = processed_dir / "val_features.parquet"
    if train_cache.exists() and val_cache.exists():
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
    else:
        require_raw_data(config)
        train_ids, val_ids, _, _ = get_patient_splits(config)
        trust = TrustAwareSignalProcessor(config)
        extractor = FeatureExtractor(config)
        labeler = OutcomeLabeler(config["data"]["outcomes_file"])
        train_df = build_split_dataset(train_ids, config, config["data"]["records_dir"], trust, extractor, labeler)
        val_df = build_split_dataset(val_ids, config, config["data"]["records_dir"], trust, extractor, labeler)
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
    feature_cols = [column for column in train_df.columns if column not in META_COLUMNS]
    return train_df, val_df, feature_cols


def json_dump(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def skipped_result(experiment: str, reason: str, prerequisites: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked_missing_data",
        "experiment": experiment,
        "reason": reason,
        "prerequisites": prerequisites,
        "baseline_preserved": True,
        "disclaimer": "No data-dependent conclusion was computed.",
    }
