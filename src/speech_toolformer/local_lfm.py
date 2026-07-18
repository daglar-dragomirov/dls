from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torchaudio


MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
TOOL_SCHEMA = (
    'You are a strict function router, not an assistant that solves requests. '
    'Your complete answer must be one JSON object and nothing else. Never explain or calculate. '
    'Use only these exact snake_case tool names: split_bill, convert_units, none. '
    'Schemas: split_bill(amount:number, people:integer, tip_percent:number, currency:"USD"|"EUR"|"RUB"); '
    'convert_units(value:number, from_unit:string, to_unit:string); none has empty arguments. '
    'Examples: "Split a USD 120 bill between 4 people with 15% tip" -> '
    '{"tool_name":"split_bill","arguments":{"amount":120,"people":4,"tip_percent":15,"currency":"USD"}}. '
    '"Convert 10 miles to kilometers" -> '
    '{"tool_name":"convert_units","arguments":{"value":10,"from_unit":"miles","to_unit":"kilometers"}}. '
    '"Tell me a joke" -> {"tool_name":"none","arguments":{}}. '
    'Start with { and end with }.'
)


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float


class LocalLFM2Audio:
    """Small local omni-model used for the official GPU benchmark."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local audio benchmark")
        self.device = torch.device(device)
        self.processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
        self.model = LFM2AudioModel.from_pretrained(model_id).eval().to(self.device)

    def _chat(self, system: str, *, audio_path: str | Path | None = None, text: str | None = None):
        from liquid_audio import ChatState

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text(system)
        chat.end_turn()
        chat.new_turn("user")
        if audio_path is not None:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            chat.add_audio(waveform, sample_rate)
            if system != "Perform ASR.":
                chat.add_text("Route the spoken request. Return the exact JSON object only.")
        elif text is not None:
            chat.add_text(text)
        else:
            raise ValueError("audio_path or text is required")
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    @torch.inference_mode()
    def _generate_text(self, chat, *, max_new_tokens: int) -> GenerationResult:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        pieces: list[str] = []
        for token in self.model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
            if token.numel() == 1:
                pieces.append(self.processor.text.decode(token))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        text = "".join(pieces).replace("<|im_end|>", "").strip()
        return GenerationResult(text, time.perf_counter() - started)

    def transcribe(self, audio_path: str | Path) -> GenerationResult:
        return self._generate_text(self._chat("Perform ASR.", audio_path=audio_path), max_new_tokens=256)

    def tool_call_from_audio(self, audio_path: str | Path, *, tuned: bool = True) -> GenerationResult:
        system = TOOL_SCHEMA if tuned else "Listen to the request and decide whether to call a tool."
        return self._generate_text(self._chat(system, audio_path=audio_path), max_new_tokens=160)

    def tool_call_from_text(self, text: str) -> GenerationResult:
        return self._generate_text(self._chat(TOOL_SCHEMA, text=text), max_new_tokens=160)
