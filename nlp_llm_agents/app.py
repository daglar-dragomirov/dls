from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data import generate_dataset
from nlp_llm_agents.experiment import run_experiment
from nlp_llm_agents.models import BaselineRelevanceModel


ARTIFACTS = ROOT / "artifacts"
BASELINE_PATH = ARTIFACTS / "baseline.joblib"


class OrganizationCard(BaseModel):
    query: str = Field(..., examples=["романтичный джаз-бар"])
    organization_name: str = Field(..., examples=["Фонарь Room"])
    category: str = Field(..., examples=["bar"])
    public_description: str = Field(..., examples=["бар коктейли уютный свет"])
    public_tags: str = Field("", examples=["bar cocktails"])
    review_snippets: str = Field("", examples=["В отзывах часто пишут: джаз, свидание, живая музыка."])
    hidden_tags: str = ""


def ensure_model() -> BaselineRelevanceModel:
    if not BASELINE_PATH.exists():
        run_experiment(train_size=4000, eval_size=300, output_dir=ARTIFACTS)
    return BaselineRelevanceModel.load(str(BASELINE_PATH))


baseline = ensure_model()
agent = RelevanceAgent(baseline)
app = FastAPI(title="DLS NLP Final: LLM Map Relevance Agent", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>LLM Map Relevance Agent</title>
      <style>
        :root { color-scheme: light; --ink:#17212b; --muted:#637083; --line:#d9e0e7; --accent:#087f5b; --warm:#f4a261; }
        * { box-sizing:border-box } body { margin:0; font-family:Inter,system-ui,sans-serif; color:var(--ink); background:#f7f9fb; }
        header { background:#12232e; color:white; padding:28px max(24px,calc((100% - 1100px)/2)); border-bottom:5px solid var(--warm); }
        header h1 { margin:0 0 6px; font-size:28px; letter-spacing:0 } header p { margin:0; color:#cad5dc }
        main { max-width:1100px; margin:24px auto; padding:0 20px 40px; display:grid; grid-template-columns:1fr 1fr; gap:20px; }
        section { background:white; border:1px solid var(--line); border-radius:8px; padding:20px; }
        .wide { grid-column:1/-1 } .metrics { display:flex; gap:12px; flex-wrap:wrap; margin-top:16px }
        .metric { min-width:145px; border-left:4px solid var(--accent); padding:8px 12px; background:#f4fbf8 }
        .metric strong { display:block; font-size:20px } .metric span { color:var(--muted); font-size:13px }
        label { display: block; margin-top: 12px; font-weight: 600; }
        input, textarea { width: 100%; box-sizing: border-box; padding: 9px; font-size: 15px; border:1px solid #bbc6d1; border-radius:5px; }
        textarea { height: 80px; }
        button { border:0; border-radius:5px; padding:10px 15px; margin:12px 8px 0 0; cursor:pointer; font-weight:700; background:var(--accent); color:white; }
        button.secondary { background:#e8eef3; color:#24313c } pre { min-height:320px; margin:0; background:#101820; color:#d8f3e8; padding:16px; overflow:auto; border-radius:6px; white-space:pre-wrap; }
        @media(max-width:780px){main{grid-template-columns:1fr}.wide{grid-column:auto}}
      </style>
    </head>
    <body><header><h1>Map Relevance Lab</h1><p>LLM agent with evidence search, guarded decisions and reproducible evaluation</p></header><main>
      <section class="wide"><h2>Experiment snapshot</h2><div class="metrics" id="metrics"></div></section><section><h2>Organization case</h2>
      <label>Query</label>
      <input id="query" value="романтичный джаз-бар"/>
      <label>Organization name</label>
      <input id="organization_name" value="Фонарь Room"/>
      <label>Category</label>
      <input id="category" value="bar"/>
      <label>Public description</label>
      <textarea id="public_description">бар коктейли уютный свет</textarea>
      <label>Public tags</label>
      <input id="public_tags" value="bar cocktails"/>
      <label>Review snippets / hidden evidence</label>
      <textarea id="review_snippets">В отзывах часто пишут: живой джаз, свидание, романтичный свет, авторские коктейли.</textarea>
      <button onclick="run()">Run agent</button><button class="secondary" onclick="negative()">Load hard negative</button></section><section><h2>Decision trace</h2><pre id="out"></pre>
      <script>
      async function run() {
        const payload = {};
        for (const id of ['query','organization_name','category','public_description','public_tags','review_snippets']) {
          payload[id] = document.getElementById(id).value;
        }
        const res = await fetch('/score', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        document.getElementById('out').textContent = JSON.stringify(await res.json(), null, 2);
      }
      run();
      fetch('/metrics').then(r=>r.json()).then(m=>{const llm=(m.real_llm_benchmark||{}).metrics||{};const values=[['Agent F1',m.agent?.f1],['Baseline F1',m.baseline?.f1],['LLM audit F1',llm.f1],['NDCG@10',m.agent?.ndcg_at_10],['Search usage',m.tool_usage_rate]];document.getElementById('metrics').innerHTML=values.map(([k,v])=>`<div class="metric"><strong>${Number(v).toFixed(3)}</strong><span>${k}</span></div>`).join('')});
      function negative(){document.getElementById('query').value='кафе куда можно с собакой';document.getElementById('organization_name').value='Север Daily';document.getElementById('category').value='cafe';document.getElementById('public_description').value='кофе и десерты, но с животными нельзя';document.getElementById('public_tags').value='cafe no_pets';document.getElementById('review_snippets').value='В отзывах подтверждают: собак и других питомцев не пускают.';run();}
      </script>
    </section></main></body>
    </html>
    """


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "nlp_llm_agents"}


@app.get("/metrics")
def metrics() -> dict:
    path = ARTIFACTS / "nlp_metrics.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    llm_path = ARTIFACTS / "nlp_llm_benchmark_metrics.json"
    if llm_path.exists():
        result["real_llm_benchmark"] = json.loads(llm_path.read_text(encoding="utf-8"))
    return result


@app.get("/demo")
def demo() -> dict:
    sample = generate_dataset(1, seed=777).iloc[0].to_dict()
    result = agent.score_one(pd.Series(sample))
    return {"input": sample, "result": result}


@app.post("/score")
def score(card: OrganizationCard) -> dict:
    row = pd.Series(card.model_dump())
    return agent.score_one(row)


@app.post("/batch_score")
def batch_score(cards: list[OrganizationCard]) -> dict:
    frame = pd.DataFrame([card.model_dump() for card in cards])
    result = agent.score_frame(frame)
    return {"items": result[["agent_score", "agent_decision", "agent_rationale", "used_search"]].to_dict(orient="records")}
