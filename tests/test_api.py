from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD_LOT = {
    "storage_days": 130,
    "temp_set_c": 2.5,
    "humidity_pct": 97,
    "power_cuts_hours": 20.4,
    "bag_weight_kg": 59.4,
    "door_openings_per_day": 18,
    "crop": "potato",
    "chamber": "Chamber 3",
}


def test_health_says_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}



def test_predict_returns_a_number():
    response = client.post("/predict", json=GOOD_LOT)
    assert response.status_code == 200
    assert isinstance(response.json()["spoilage_pct"], float)



def test_bag_weight_above_85_is_rejected():
    bad_lot = {**GOOD_LOT, "bag_weight_kg": 86}
    response = client.post("/predict", json=bad_lot)
    assert response.status_code == 422


def test_negative_door_openings_is_rejected():
    bad_lot = {**GOOD_LOT, "door_openings_per_day": -1}
    response = client.post("/predict", json=bad_lot)
    assert response.status_code == 422


def test_limits_themselves_are_allowed():
    edge_lot = {**GOOD_LOT, "bag_weight_kg": 85, "door_openings_per_day": 0}
    response = client.post("/predict", json=edge_lot)
    assert response.status_code == 200