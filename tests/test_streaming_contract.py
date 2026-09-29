"""Contract tests for the local incremental inference API models."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.api.router import ObservationBatchRequest
from backend.main import app
from ml.evidence_accumulator import SequentialEvidenceAccumulator
from ml.preprocessing import load_config


def test_observation_batch_requires_non_empty_observations():
    with pytest.raises(ValidationError):
        ObservationBatchRequest(timestamp_hours=1.0, observations={})


def test_observation_batch_rejects_out_of_horizon_time():
    with pytest.raises(ValidationError):
        ObservationBatchRequest(timestamp_hours=49.0, observations={"HR": 80.0})


def test_observation_batch_accepts_static_context():
    request = ObservationBatchRequest(
        timestamp_hours=4.0,
        observations={"HR": 80.0, "SysABP": 120.0},
        static_info={"Age": 65, "ICUType": 2},
    )
    assert request.observations["HR"] == 80.0
    assert request.static_info["Age"] == 65


def test_alert_budget_and_refractory_are_enforced():
    accumulator = SequentialEvidenceAccumulator(load_config("configs/config.yaml"))
    first = accumulator.update(1.0, 0.75, 0.75, credibility_weight=1.0)
    second = accumulator.update(2.0, 0.85, 0.85, credibility_weight=1.0)
    third = accumulator.update(3.0, 0.90, 0.90, credibility_weight=1.0)
    assert first.state in {"WATCH", "ALERT"}
    assert second.alert_fired
    assert not third.alert_fired
    assert third.suppressed_by_refractory


def test_health_version_and_incremental_api_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/version").json()["service"] == "silentwindow"

    patient_id = 991234
    first = client.post(
        f"/api/v1/patients/{patient_id}/observations",
        json={"timestamp_hours": 2.0, "observations": {"HR": 80.0, "SysABP": 120.0}},
    )
    assert first.status_code == 200, first.text
    assert first.json()["state"] == "STABLE"

    out_of_order = client.post(
        f"/api/v1/patients/{patient_id}/observations",
        json={"timestamp_hours": 1.0, "observations": {"HR": 81.0}},
    )
    assert out_of_order.status_code == 400
    assert client.get(f"/api/v1/patients/{patient_id}/state").status_code == 200
    assert client.delete(f"/api/v1/patients/{patient_id}/state").json()["reset"] is True


def test_policy_sweep_returns_operating_points():
    response = TestClient(app).get("/api/policy-sweep?thresholds=0.65,0.75")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["results"]) == 2
    assert "alerts_per_patient_day" in payload["results"][0]
