import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


client = TestClient(app)


def test_health_ui_metrics_and_score_end_to_end():
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "Агент релевантности" in page
    assert "Готово: класс" in page
    assert "scrollIntoView" in page
    assert "official_eval" in client.get("/metrics").json()
    response = client.post(
        "/score",
        json={
            "query": "ресторан с верандой",
            "organization_name": "Северный сад",
            "category": "Ресторан",
            "review_snippets": "Гости хвалят просторную летнюю веранду.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_relevance"] in [0.0, 0.1, 1.0]
    assert len(body["tool_calls"]) == 2
