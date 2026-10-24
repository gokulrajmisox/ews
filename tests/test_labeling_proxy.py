"""
test_labeling_proxy.py - Regression tests for the configurable training target.

Covers ml/labeling.py's OutcomeLabeler.assign_training_labels() (the Day 10
change on branch day10-proxy-target):

- "future_window_proxy" (default): an hourly row is positive only when the
  patient dies in hospital AND 0 < (endpoint_proxy - eval_time) <= horizon.
- "eventual_outcome" (legacy): the patient's final in-hospital outcome
  broadcast to every hourly row (identical to assign_hourly_labels).
- Error paths: missing event_proxy_hours, non-positive horizon, missing
  eval_time column, unknown target kind.
- Meta-column contract: the labeler only ever attaches the documented
  label/meta columns. The string disclaimer column in particular must stay
  out of the XGBoost feature set (excluded via meta_cols in ml/train.py).
"""

import os

import pandas as pd
import pytest

from ml.labeling import OutcomeLabeler

OUTCOMES_CSV = """RecordID,In-hospital_death,SOFA,SAPS-I,Length_of_stay,Survival
1,1,9.0,62.0,15.2,11.5
2,0,4.0,38.0,7.8,190.0
"""


@pytest.fixture
def labeler(tmp_path):
    outcomes_file = tmp_path / "Outcomes-train.txt"
    outcomes_file.write_text(OUTCOMES_CSV)
    return OutcomeLabeler(outcomes_file=str(outcomes_file))


@pytest.fixture
def features():
    return pd.DataFrame(
        {
            "eval_time": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
            "HR_mean": [80.0, 82.0, 85.0, 90.0, 95.0, 100.0],
        }
    )


def test_proxy_positives_only_within_horizon(labeler, features):
    """Dying patient: only rows with 0 < endpoint - eval_time <= horizon are 1."""
    df = labeler.assign_training_labels(
        features, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
    )
    # deltas = 20 - eval_time = [20, 16, 12, 8, 4, 0];
    # positive iff 0 < delta <= 12 -> eval_time in {8, 12, 16}.
    # Boundary: delta == 12 counts (<=), delta == 0 does not (> 0 strictly).
    assert df["future_window_proxy_target"].tolist() == [0, 0, 1, 1, 1, 0]


def test_proxy_survivor_all_zero(labeler, features):
    df = labeler.assign_training_labels(
        features, patient_id=2, event_proxy_hours=20.0, horizon_hours=12.0
    )
    assert df["future_window_proxy_target"].tolist() == [0, 0, 0, 0, 0, 0]
    assert df["target"].tolist() == [0, 0, 0, 0, 0, 0]


def test_proxy_target_alias_matches_label(labeler, features):
    """The stable `target` column must always equal the proxy label column."""
    df = labeler.assign_training_labels(
        features, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
    )
    assert (df["target"] == df["future_window_proxy_target"]).all()


def test_proxy_default_horizon_is_12h(labeler, features):
    """Default args must match configs/config.yaml: horizon 12.0 hours."""
    df = labeler.assign_training_labels(features, patient_id=1, event_proxy_hours=20.0)
    assert (df["future_window_horizon_hours"] == 12.0).all()
    assert df["future_window_proxy_target"].tolist() == [0, 0, 1, 1, 1, 0]


def test_proxy_attaches_outcome_metadata(labeler, features):
    df = labeler.assign_training_labels(
        features, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
    )
    assert (df["sofa"] == 9.0).all()
    assert (df["saps"] == 62.0).all()
    assert (df["length_of_stay"] == 15.2).all()


def test_legacy_broadcasts_final_outcome(labeler, features):
    df = labeler.assign_training_labels(
        features, patient_id=1, target_kind="eventual_outcome"
    )
    assert df["target"].tolist() == [1, 1, 1, 1, 1, 1]
    df = labeler.assign_training_labels(
        features, patient_id=2, target_kind="eventual_outcome"
    )
    assert df["target"].tolist() == [0, 0, 0, 0, 0, 0]


def test_legacy_matches_assign_hourly_labels(labeler, features):
    """Legacy mode must be byte-equivalent to the pre-Day-10 code path."""
    via_new = labeler.assign_training_labels(
        features, patient_id=1, target_kind="eventual_outcome"
    )
    via_old = labeler.assign_hourly_labels(features, patient_id=1)
    pd.testing.assert_frame_equal(via_new, via_old)


def test_proxy_requires_event_proxy_hours(labeler, features):
    with pytest.raises(ValueError, match="event_proxy_hours"):
        labeler.assign_training_labels(features, patient_id=1)


def test_proxy_rejects_nonpositive_horizon(labeler, features):
    with pytest.raises(ValueError, match="horizon_hours"):
        labeler.assign_training_labels(
            features, patient_id=1, event_proxy_hours=20.0, horizon_hours=0.0
        )


def test_proxy_requires_eval_time_column(labeler):
    df = pd.DataFrame({"HR_mean": [80.0]})
    with pytest.raises(ValueError, match="eval_time"):
        labeler.assign_training_labels(
            df, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
        )


def test_unknown_target_kind_raises(labeler, features):
    with pytest.raises(ValueError, match="Unknown labeling target_kind"):
        labeler.assign_training_labels(
            features, patient_id=1, target_kind="next_week", event_proxy_hours=20.0
        )


def test_proxy_meta_column_contract(labeler, features):
    """New columns in proxy mode are exactly the documented label/meta set.

    Anything else leaking in here would either break the train path or, in
    the case of the string disclaimer, must be kept out of the XGBoost
    feature set (see meta_cols in ml/train.py).
    """
    before = set(features.columns)
    df = labeler.assign_training_labels(
        features, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
    )
    new_cols = set(df.columns) - before
    assert new_cols == {
        "future_window_proxy_target",
        "future_window_horizon_hours",
        "future_window_proxy_disclaimer",
        "target",
        "sofa",
        "saps",
        "length_of_stay",
    }
    # The disclaimer is a string column (pandas 2.x: object dtype, 3.x: str
    # dtype): this is why meta_cols must exclude it from XGBoost features.
    assert pd.api.types.is_string_dtype(df["future_window_proxy_disclaimer"])
    assert "unavailable" in df["future_window_proxy_disclaimer"].iloc[0]


def test_legacy_meta_column_contract(labeler, features):
    before = set(features.columns)
    df = labeler.assign_training_labels(
        features, patient_id=1, target_kind="eventual_outcome"
    )
    new_cols = set(df.columns) - before
    assert new_cols == {"target", "sofa", "saps", "length_of_stay"}


def test_input_frame_not_mutated(labeler, features):
    before = features.copy()
    labeler.assign_training_labels(
        features, patient_id=1, event_proxy_hours=20.0, horizon_hours=12.0
    )
    pd.testing.assert_frame_equal(features, before)


def test_labeling_config_defaults_are_proxy(tmp_path):
    """Repo config must default to the proxy target (configs/config.yaml).

    Fails loudly if someone flips the default back to eventual_outcome.
    """
    import yaml

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(repo_root, "configs", "config.yaml")) as f:
        config = yaml.safe_load(f)
    assert config["labeling"]["target"] == "future_window_proxy"
    assert config["labeling"]["horizon_hours"] == 12.0
