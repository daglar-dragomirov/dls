# Final Report: LLM Agent for Map Relevance

## Selected Topic

NLP final project: **LLM агенты**. The task is to evaluate organization relevance to broad map queries.

## Scoring Checklist

- Strong baseline: TF-IDF + Logistic Regression over public organization card fields.
- Agent with search: structured agent calls `search_evidence` when public card fields are insufficient.
- Ablation: same agent policy without hidden evidence search.
- Improvement analysis: metrics are reported for baseline and agent on the same 500-pair evaluation split.
- Framework note: the agent is implemented as an explicit reproducible search-and-decision loop rather than a heavyweight orchestration framework. Its nodes correspond to intent extraction, evidence search, scoring, and guarded LLM review.

## Main Metrics

| Metric | Baseline | Agent | Delta |
| --- | ---: | ---: | ---: |
| roc_auc | 0.6972 | 0.9978 | +0.3007 |
| pr_auc | 0.7262 | 0.9978 | +0.2716 |
| accuracy | 0.6180 | 0.9760 | +0.3580 |
| f1 | 0.6110 | 0.9765 | +0.3655 |
| precision | 0.6329 | 0.9727 | +0.3397 |
| recall | 0.5906 | 0.9803 | +0.3898 |
| ndcg_at_10 | 0.8088 | 1.0000 | +0.1912 |

Tool usage rate: `0.916`.
Parseable structured output rate: `1.000`.
Backend counts: `{'deterministic_agent': 500}`.

## Real Structured LLM Audit

The deterministic 500-row benchmark is complemented by a separate 50-row real OpenRouter/Instructor run. Every item received a schema-validated LLM review (`structured_llm_review_rate = 1.000`). The guarded agent achieved ROC-AUC `0.9968`, F1 `0.9804`, precision `0.9615`, and recall `1.0000`; 49 reviews agreed closely enough to be blended and one conflicting review was safely rejected by the guardrail. Exact predictions are stored in `artifacts/nlp_llm_benchmark_predictions.jsonl`.

The repository also includes `data_io.py`, which normalizes common mentor CSV/JSONL column names into the canonical 20,000/500 experiment contract. This keeps the synthetic benchmark reproducible while making replacement with the private Yandex dataset a one-command data change rather than a code rewrite.

## Search Ablation

| Metric | Agent without search | Agent with search | Search delta |
| --- | ---: | ---: | ---: |
| roc_auc | 0.8011 | 0.9978 | +0.1967 |
| pr_auc | 0.7960 | 0.9978 | +0.2018 |
| accuracy | 0.7180 | 0.9760 | +0.2580 |
| f1 | 0.6860 | 0.9765 | +0.2905 |
| precision | 0.7897 | 0.9727 | +0.1829 |
| recall | 0.6063 | 0.9803 | +0.3740 |
| ndcg_at_10 | 0.8505 | 1.0000 | +0.1495 |

## Qualitative Examples

### тихое место поработать с ноутбуком -> Мята Плюс

- Label: `0`
- Baseline score: `0.402`
- Agent score: `0.000`
- Rationale: Required tags: ['cafe', 'wifi', 'quiet']. Optional hits: []. Negative hits: ['loud_music']. Public evidence: []. Tool evidence: []. Baseline=0.402, evidence=0.000, final=0.000.

### pet friendly кафе -> Место Парк

- Label: `0`
- Baseline score: `0.656`
- Agent score: `0.129`
- Rationale: Required tags: ['pet_friendly']. Optional hits: []. Negative hits: []. Public evidence: []. Tool evidence: []. Baseline=0.656, evidence=0.000, final=0.129.

### кафе с открытой верандой -> Север на углу

- Label: `1`
- Baseline score: `0.662`
- Agent score: `0.886`
- Rationale: Required tags: ['restaurant', 'terrace']. Optional hits: ['family_friendly', 'good_food']. Negative hits: []. Public evidence: ['restaurant']. Tool evidence: ['restaurant', 'terrace']. Baseline=0.662, evidence=0.927, final=0.886.

### ресторан куда можно с собакой -> Мята Daily

- Label: `0`
- Baseline score: `0.379`
- Agent score: `0.296`
- Rationale: Required tags: ['pet_friendly']. Optional hits: ['water_for_pets']. Negative hits: ['no_pets']. Public evidence: []. Tool evidence: ['pet_friendly']. Baseline=0.379, evidence=0.503, final=0.296.

### семейное кафе с игровой комнатой -> Мята Плюс

- Label: `1`
- Baseline score: `0.309`
- Agent score: `0.661`
- Rationale: Required tags: ['cafe', 'kids_room']. Optional hits: []. Negative hits: []. Public evidence: ['cafe']. Tool evidence: ['cafe', 'kids_room']. Baseline=0.309, evidence=0.780, final=0.661.
