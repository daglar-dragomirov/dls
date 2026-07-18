import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


client = TestClient(app)


def test_health_ui_metrics_and_text_tool_end_to_end():
    assert client.get("/health").json()["status"] == "ok"
    assert "Голосовой" in client.get("/").text
    assert "local_lfm" in client.get("/metrics").json()
    response = client.post("/assistant/text", json={"text": "Convert 10 miles to kilometers"})
    assert response.status_code == 200
    body = response.json()
    assert body["call"]["tool_name"] == "convert_units"
    assert body["valid"] is True


def test_sample_audio_is_downloadable():
    response = client.get("/sample-audio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 1_000
