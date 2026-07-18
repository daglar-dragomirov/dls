from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.audio_backend import direct_audio_tool_call, text_llm_tool_call, transcribe_audio
from speech_toolformer.experiment import run_experiment
from speech_toolformer.pipeline import cascaded_audio_pipeline, text_pipeline
from speech_toolformer.tools import execute_tool


class TextRequest(BaseModel):
    text: str


app = FastAPI(title="DLS Speech: голосовой вызов инструментов", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "speech_toolformer", "version": "2.0.0"}


@app.get("/metrics")
def metrics() -> dict:
    path = ROOT / "artifacts" / "speech_metrics.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for key, filename in [("real_audio", "real_audio_metrics.json"), ("local_lfm", "local_lfm_metrics.json")]:
        artifact = ROOT / "artifacts" / filename
        if artifact.exists():
            result[key] = json.loads(artifact.read_text(encoding="utf-8"))
    return result


@app.get("/sample-audio")
def sample_audio() -> FileResponse:
    path = ROOT / "artifacts" / "audio_samples" / "0000.mp3"
    return FileResponse(path, media_type="audio/mpeg", filename="speech_toolformer_test_en.mp3")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return r"""
<!doctype html><html lang="ru"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Голосовой ассистент Speech-Toolformer</title><style>
:root{--ink:#17212b;--muted:#637083;--line:#d9e0e7;--accent:#c2410c;--cyan:#0e7490;--paper:#fff;--bg:#f5f7f9}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--ink);background:var(--bg);letter-spacing:0}header{padding:25px max(20px,calc((100% - 1080px)/2));background:#17212b;color:#fff;border-bottom:4px solid #f59e0b}h1{margin:0 0 6px;font-size:27px}header p{margin:0;color:#cbd5e1}
main{max-width:1080px;margin:22px auto;padding:0 18px 38px;display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:19px}.wide{grid-column:1/-1}.metrics{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:11px}.metric{border-left:4px solid var(--cyan);padding:7px 10px;background:#f0f9fb}.metric strong{display:block;font-size:20px}.metric span{font-size:12px;color:var(--muted)}h2{font-size:18px;margin:0 0 12px}p.hint{font-size:13px;color:var(--muted);line-height:1.5}
textarea,input,select{width:100%;padding:10px;border:1px solid #bbc6d1;border-radius:5px;font:inherit;margin:6px 0}textarea{min-height:92px;resize:vertical}button,.download{display:inline-block;border:0;border-radius:5px;padding:10px 14px;margin:8px 7px 0 0;background:var(--accent);color:#fff;font-weight:700;cursor:pointer;text-decoration:none}.download{background:#e7edf0;color:#263842}pre{min-height:390px;max-height:620px;margin:0;background:#101820;color:#d8f3e8;padding:15px;overflow:auto;border-radius:6px;white-space:pre-wrap;font-size:13px}.status{font-size:13px;color:var(--muted);margin-top:8px}@media(max-width:800px){main{grid-template-columns:1fr}.wide{grid-column:auto}.metrics{grid-template-columns:1fr 1fr}}
</style></head><body><header><h1>Голосовой ассистент Speech-Toolformer</h1><p>Аудиозапросы, JSON-вызовы функций и сравнение прямого и каскадного пайплайнов</p></header>
<main><section class="panel wide"><h2>Результаты эксперимента</h2><div id="metrics" class="metrics"></div></section>
<section class="panel"><h2>Проверка ассистента</h2><p class="hint">Локальная LFM2.5-Audio оценивалась на английской речи, поэтому готовый тестовый файл содержит английскую команду. Интерфейс и пояснения доступны на русском.</p>
<label for="text">Текстовый запрос</label><textarea id="text">Split a USD 120 bill between 4 people with 15% tip</textarea><button onclick="runText()">Выполнить текстовый запрос</button>
<h2 style="margin-top:22px">Аудиофайл</h2><a class="download" href="/sample-audio">Скачать тестовый MP3</a><input id="audio" type="file" accept="audio/*"/>
<select id="pipeline"><option value="direct">Прямой: аудио → вызов функции</option><option value="cascaded">Каскадный: ASR → текстовая модель → вызов функции</option></select><button onclick="runAudio()">Обработать аудио</button><div class="status" id="status"></div></section>
<section class="panel"><h2>Результат и трасса пайплайна</h2><pre id="out">Загрузка…</pre></section></main>
<script>
async function runText(){const status=document.getElementById('status'),out=document.getElementById('out'),text=document.getElementById('text');status.textContent='Выполняется…';const response=await fetch('/assistant/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value})});out.textContent=JSON.stringify(await response.json(),null,2);status.textContent=response.ok?'Готово':'Ошибка запроса';}
async function runAudio(){const status=document.getElementById('status'),out=document.getElementById('out'),audio=document.getElementById('audio'),pipeline=document.getElementById('pipeline');const file=audio.files[0];if(!file){out.textContent='Сначала выберите аудиофайл.';return}status.textContent='Аудио обрабатывается…';const form=new FormData();form.append('audio',file);form.append('pipeline',pipeline.value);const response=await fetch('/assistant/audio',{method:'POST',body:form});out.textContent=JSON.stringify(await response.json(),null,2);status.textContent=response.ok?'Готово':'Ошибка обработки';}
fetch('/metrics').then(r=>r.json()).then(m=>{const t=m.text_pipeline||{},l=m.local_lfm||m.real_audio||{},c=l.pipeline_C_direct_audio||l.pipeline_C_native_audio||{},d=l.pipeline_D_cascaded||{},b=l.pipeline_B_asr||{};const values=[['F1 на тексте',t.f1],['F1 прямого пайплайна',c.f1],['Аргументы C',c.argument_accuracy],['Аргументы D',d.argument_accuracy],['WER локальной ASR',b.wer]];document.getElementById('metrics').innerHTML=values.map(([k,v])=>`<div class="metric"><strong>${Number(v||0).toFixed(3)}</strong><span>${k}</span></div>`).join('')});runText();
</script></body></html>"""


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
        raise HTTPException(status_code=400, detail="Пустой аудиофайл")
    suffix = Path(audio.filename or "audio.wav").suffix.lower().lstrip(".") or "wav"
    if suffix not in {"wav", "mp3", "m4a", "ogg", "flac", "aac"}:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: {suffix}")
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
        raise HTTPException(status_code=400, detail="pipeline должен быть direct или cascaded")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ошибка аудиобэкенда: {type(exc).__name__}") from exc


@app.post("/experiment")
def experiment() -> dict:
    return run_experiment(n=300)
