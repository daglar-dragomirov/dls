import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module


client = TestClient(app_module.app)


def test_health_ui_metrics_and_text_tool_end_to_end():
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "Голосовой" in page
    assert "Выбранная тема DLS" in page
    assert "Готово: ${tool}" in page
    assert "scrollIntoView" in page
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


def test_direct_audio_executes_the_validated_tool(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "direct_audio_tool_call",
        lambda content, suffix: {
            "raw": "{}",
            "call": {
                "tool_name": "split_bill",
                "arguments": {
                    "amount": 120,
                    "people": 6,
                    "tip_percent": 18,
                    "currency": "USD",
                },
            },
            "valid": True,
            "reason": "ok",
        },
    )
    response = client.post(
        "/assistant/audio",
        files={"audio": ("sample.mp3", b"test audio", "audio/mpeg")},
        data={"pipeline": "direct"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "C_native_audio"
    assert body["tool_result"] == {"total": 141.6, "per_person": 23.6, "currency": "USD"}
    assert "23.6 USD" in body["answer"]
