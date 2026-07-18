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
    assert "Релевантность организаций" in page
    assert "Выбранная тема DLS" in page
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
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_relevance"] in [0.0, 0.1, 1.0]
    assert len(body["tool_calls"]) == 2


def test_validation_and_batch_modes(monkeypatch):
    import app as module
    assert client.post('/score', json={'query': '  ', 'organization_name': 'x'}).status_code == 422
    assert client.post('/score', json={'query': 'x', 'organization_name': 'x', 'search_mode': 'bad'}).status_code == 422
    calls = []
    monkeypatch.setattr(module, 'instructor_available', lambda: True)
    def fake_score(row, **kwargs):
        calls.append(kwargs)
        return {'backend': 'test'}
    monkeypatch.setattr(module.agent, 'score_one', fake_score)
    card = {'query': 'x', 'organization_name': 'y'}
    response = client.post('/batch_score', json=[card, {**card, 'use_llm': False}, {**card, 'use_web_search': True}])
    assert response.status_code == 200
    assert calls == [dict(use_llm=True, search_mode='auto'), dict(use_llm=False, search_mode='off'), dict(use_llm=True, search_mode='always')]
    assert client.post('/batch_score', json=[card] * 11).status_code == 422


def test_llm_failure_is_not_silently_replaced(monkeypatch):
    import app as module
    from nlp_llm_agents.instructor_backend import LLMUnavailable
    card = {'query': 'x', 'organization_name': 'y'}
    monkeypatch.setattr(module, 'instructor_available', lambda: False)
    assert client.post('/score', json=card).status_code == 503
    monkeypatch.setattr(module, 'instructor_available', lambda: True)
    def fail(*args, **kwargs):
        raise LLMUnavailable('Provider unavailable')
    monkeypatch.setattr(module.agent, 'score_one', fail)
    assert client.post('/score', json=card).status_code == 502


def test_paid_request_limits_do_not_lock_baseline(monkeypatch):
    import app as module
    monkeypatch.setattr(module, 'instructor_available', lambda: True)
    monkeypatch.setenv('LLM_REQUESTS_PER_HOUR', '0')
    card = {'query': 'restaurant', 'organization_name': 'place'}
    assert client.post('/score', json=card).status_code == 429
    monkeypatch.setattr(module.agent, 'score_one', lambda *args, **kwargs: {'ok': True})
    assert client.post('/score', json={**card, 'use_llm': False}).status_code == 200
    monkeypatch.setenv('LLM_REQUESTS_PER_HOUR', '100')
    assert client.post('/score', json=card).status_code == 200
