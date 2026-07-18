from __future__ import annotations

import random


class TranscriptASR:
    """Reproducible ASR simulator for generated examples.

    The project exposes the same interface a real ASR module would have. In the
    experiment we use a deterministic corruption model to measure WER and the
    text-vs-audio modality gap without depending on paid APIs.
    """

    def __init__(self, word_drop_prob: float = 0.025, seed: int = 2026):
        self.word_drop_prob = word_drop_prob
        self.rng = random.Random(seed)

    def transcribe(self, transcript: str) -> str:
        words = transcript.split()
        if not words:
            return transcript
        kept = [word for word in words if self.rng.random() > self.word_drop_prob]
        return " ".join(kept or words[:1])


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i, rw in enumerate(ref, 1):
        for j, hw in enumerate(hyp, 1):
            cost = 0 if rw == hw else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(ref)

