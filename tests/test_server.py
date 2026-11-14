"""End-to-end HTTP tests against the FastAPI app."""
from fastapi.testclient import TestClient

from hoa_env.server.app import app

client = TestClient(app)


def test_health_and_root():
    assert client.get("/health").json()["status"] == "healthy"
    assert "Heuristic Override Arena" in client.get("/").text


def test_info_counts_match_data():
    info = client.get("/info").json()
    tasks = client.get("/tasks").json()
    assert info["total_scenarios"] == sum(t["count"] for t in tasks.values()) == 100
    assert info["domains"] == len(tasks) == 5
    all_types = {h for t in tasks.values() for h in t["heuristic_types"]}
    assert info["bias_types"] == len(all_types)


def test_http_reset_step_roundtrip():
    r = client.post("/reset", json={"task": "procurement", "seed": 1}).json()
    scenario = r["observation"]["scenario"]
    assert "ground_truth" not in scenario
    from hoa_env.environment import HOAEnvironment

    gt = next(
        s["ground_truth"]
        for s in HOAEnvironment()._scenarios["procurement"]
        if s["id"] == scenario["id"]
    )
    s = client.post("/step", json={"action": {"choice": gt["correct_choice"]}}).json()
    assert s["reward"] >= 0.6
    assert s["done"] is False
