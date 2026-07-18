import pandas as pd
import pytest

from nlp_llm_agents import agent as module
from nlp_llm_agents.schemas import RelevanceDecision, SearchPlan
from nlp_llm_agents.instructor_backend import LLMUnavailable


class Baseline:
    def predict_proba(self, frame):
        import numpy as np
        return np.tile([.7, .2, .1], (len(frame), 1))


@pytest.mark.parametrize('mode,needs_search,expected', [('auto', True, True), ('auto', False, False), ('off', True, False), ('always', False, True)])
def test_planning_search_and_evidence_reach_decision(monkeypatch, mode, needs_search, expected):
    captured = []
    searches = []
    monkeypatch.setattr(module, 'plan_search', lambda payload: SearchPlan(needs_search=needs_search, query='place city terrace', reason='Недостаточно фактов'))
    def search(query):
        searches.append(query)
        return {'status': 'ok', 'results': [{'url': 'https://example.org', 'title': 'Terrace'}], 'summary': 'Есть веранда'}
    monkeypatch.setattr(module, 'search_web', search)
    def decide(payload):
        captured.append(payload)
        return RelevanceDecision(relevance=1.0, confidence=.8, evidence_summary='Веранда подтверждена', rationale='Подходит')
    monkeypatch.setattr(module, 'try_instructor_decision', decide)
    row = pd.Series({'query': 'terrace', 'organization_name': 'place', 'address': 'city', 'review_snippets': 'x' * 900 + ' tail evidence'})
    result = module.RelevanceAgent(Baseline()).score_one(row, use_llm=True, search_mode=mode)
    assert bool(searches) == expected
    assert result['used_web_search'] == expected
    assert captured[0]['organization_card']['reviews'].endswith('tail evidence')
    assert (captured[0]['public_web_search'] is not None) == expected
    assert result['backend'] == 'structured_llm_agent'
    assert result['predicted_relevance'] == 1.0
    assert result['probability_1.0'] is None
    assert result['confidence_kind'] == 'llm_self_assessment'


def test_missing_sources_preserved_and_failure_propagates(monkeypatch):
    monkeypatch.setattr(module, 'search_web', lambda query: {'status': 'unavailable', 'results': [], 'summary': ''})
    def decide(payload):
        assert payload['public_web_search']['status'] == 'unavailable'
        raise LLMUnavailable('Unavailable')
    monkeypatch.setattr(module, 'try_instructor_decision', decide)
    with pytest.raises(LLMUnavailable):
        module.RelevanceAgent(Baseline()).score_one(pd.Series({'query': 'x'}), use_llm=True, search_mode='always')


@pytest.mark.parametrize('verdict,label', [('IRRELEVANT', 0.0), ('PARTIAL', 0.1), ('RELEVANT', 1.0)])
def test_semantic_classes_map_to_official_labels(monkeypatch, verdict, label):
    from nlp_llm_agents import instructor_backend as backend
    from nlp_llm_agents.schemas import SemanticDecision
    monkeypatch.setattr(backend, '_structured', lambda *args: SemanticDecision(verdict=verdict, confidence=.8, rationale='Reason', evidence_summary='Evidence'))
    assert backend.try_instructor_decision({}).relevance == label


def test_search_plan_requires_a_query_when_searching():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SearchPlan(needs_search=True, query=' ', reason='Missing information')
    assert not SearchPlan(needs_search=False, query='', reason='Clear match').needs_search
