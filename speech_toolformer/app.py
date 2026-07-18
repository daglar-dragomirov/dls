from __future__ import annotations

from pathlib import Path
import sys
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.experiment import run_experiment
from speech_toolformer.audio_backend import direct_audio_tool_call, text_llm_tool_call, transcribe_audio
from speech_toolformer.pipeline import cascaded_audio_pipeline, text_pipeline
from speech_toolformer.tools import execute_tool


class TextRequest(BaseModel):
    text: str


app = FastAPI(title="DLS Speech Final: Speech-Toolformer", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "speech_toolformer"}


@app.get("/metrics")
def metrics() -> dict:
    path = ROOT / "artifacts" / "speech_metrics.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    real_path = ROOT / "artifacts" / "real_audio_metrics.json"
    if real_path.exists():
        result["real_audio"] = json.loads(real_path.read_text(encoding="utf-8"))
    return result


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Speech-Toolformer</title>
      <style>
        :root{--ink:#17212b;--muted:#637083;--line:#d9e0e7;--accent:#c2410c;--cyan:#0e7490}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--ink);background:#f8fafc}header{padding:28px max(24px,calc((100% - 1080px)/2));background:#17212b;color:white;border-bottom:5px solid #f59e0b}header h1{margin:0 0 6px;font-size:28px;letter-spacing:0}header p{margin:0;color:#cbd5e1}main{max-width:1080px;margin:24px auto;padding:0 20px 40px;display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;border:1px solid var(--line);border-radius:8px;padding:20px}.wide{grid-column:1/-1}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{min-width:150px;border-left:4px solid var(--cyan);padding:8px 12px;background:#f0f9fb}.metric strong{display:block;font-size:20px}.metric span{font-size:13px;color:var(--muted)}textarea,input,select{width:100%;padding:10px;border:1px solid #bbc6d1;border-radius:5px;font:inherit;margin:6px 0}textarea{min-height:100px}button{border:0;border-radius:5px;padding:10px 15px;margin:8px 8px 0 0;background:var(--accent);color:white;font-weight:700;cursor:pointer}pre{min-height:330px;margin:0;background:#101820;color:#d8f3e8;padding:16px;overflow:auto;border-radius:6px;white-space:pre-wrap}@media(max-width:780px){main{grid-template-columns:1fr}.wide{grid-column:auto}}
      </style>
    </head>
    <body><header><h1>Speech-Toolformer Lab</h1><p>Real bilingual audio, structured calls and side-by-side pipeline evaluation</p></header><main><section class="wide"><h2>Benchmark snapshot</h2><div id="metrics" class="metrics"></div></section><section><h2>Try the assistant</h2>
      <textarea id="text">Split a USD 120 bill between 4 people with 15% tip</textarea>
      <br/>
      <button onclick="run()">Run assistant</button>
      <h2>Real audio</h2>
      <input id="audio" type="file" accept="audio/*"/>
      <select id="pipeline"><option value="direct">Native audio → tool call</option><option value="cascaded">ASR → text LLM → tool call</option></select>
      <button onclick="runAudio()">Process audio</button>
      </section><section><h2>Pipeline output</h2><pre id="out"></pre>
      <script>
      async function run() {
        const text = document.getElementById('text').value;
        const res = await fetch('/assistant/text', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text})
        });
        document.getElementById('out').textContent = JSON.stringify(await res.json(), null, 2);
      }
      run();
      fetch('/metrics').then(r=>r.json()).then(m=>{const t=m.text_pipeline||{},r=m.real_audio||{},c=r.pipeline_C_native_audio||{},d=r.pipeline_D_cascaded||{};const vals=[['Text F1',t.f1],['Native C F1',c.f1],['C args',c.argument_accuracy],['D args',d.argument_accuracy],['Real ASR WER',(r.pipeline_B_asr||{}).wer]];document.getElementById('metrics').innerHTML=vals.map(([k,v])=>`<div class="metric"><strong>${Number(v).toFixed(3)}</strong><span>${k}</span></div>`).join('')});
      async function runAudio() {
        const file = document.getElementById('audio').files[0];
        if (!file) { document.getElementById('out').textContent = 'Choose an audio file first.'; return; }
        const form = new FormData();
        form.append('audio', file);
        form.append('pipeline', document.getElementById('pipeline').value);
        const res = await fetch('/assistant/audio', {method: 'POST', body: form});
        document.getElementById('out').textContent = JSON.stringify(await res.json(), null, 2);
      }
      </script>
    </section></main></body>
    </html>
    """


@app.get("/demo")
def demo() -> dict:
    return text_pipeline("Convert 10 kilometers to miles")


@app.post("/assistant/text")
def assistant_text(request: TextRequest) -> dict:
    return text_pipeline(request.text)


@app.post("/assistant/audio_transcript")
def assistant_audio_transcript(request: TextRequest) -> dict:
    return cascaded_audio_pipeline(request.text)


@app.post("/assistant/audio")
async def assistant_audio(audio: UploadFile = File(...), pipeline: str = Form("direct")) -> dict:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    suffix = Path(audio.filename or "audio.wav").suffix.lower().lstrip(".") or "wav"
    if suffix not in {"wav", "mp3", "m4a", "ogg", "flac", "aac"}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix}")
    try:
        if pipeline == "direct":
            result = direct_audio_tool_call(content, suffix)
            result["pipeline"] = "C_native_audio"
            return result
        if pipeline == "cascaded":
            transcript = transcribe_audio(content, suffix)
            result = text_llm_tool_call(transcript["text"])
            result.update({"pipeline": "D_cascaded", "recognized_text": transcript["text"], "stt_usage": transcript["usage"]})
            if result.get("valid") and result.get("call", {}).get("tool_name") != "none":
                executed = execute_tool(result["call"])
                result.update({"tool_result": executed.result, "answer": executed.answer})
            return result
        raise HTTPException(status_code=400, detail="pipeline must be direct or cascaded")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio backend failed: {type(exc).__name__}") from exc


@app.post("/experiment")
def experiment() -> dict:
    return run_experiment(n=300)
