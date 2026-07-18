# Five-minute defense

1. Open `/` and show the real A/B/C/D benchmark cards.
2. Run a text request and point to the validated JSON call and executed result.
3. Upload any file from `artifacts/audio_samples`, first with native C and then cascaded D.
4. Compare latency: C averages 1.51 s, D 3.02 s; both reach F1 1.00 on the fixed real-audio subset.
5. Open `reports/final_report.md` and show the zero-shot vs schema-tuned parseability jump, real WER, modality comparison, and failure analysis.

Key conclusion: prompt/schema tuning is sufficient for the scoped tools; native audio is the best online route, while cascaded ASR remains the most interpretable debugging route.
