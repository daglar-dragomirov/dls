# DLS Final Projects

This repository contains two independent final projects selected from the DLS Spring 2026 project sheets.

## Selected Topics

| Track | Selected topic | Why this topic |
| --- | --- | --- |
| NLP | LLM agents for organization-query relevance in maps | Non-partner topic with clear scoring: baseline, agent with search, and measurable improvement. |
| Speech | Speech-Toolformer | Non-selection topic; can be delivered as a reproducible assistant with tool calls, synthetic data, metrics, and an HTTP service. |

## Projects

- `nlp_llm_agents/` - relevance scoring baseline and tool-using agent.
- `speech_toolformer/` - voice/text assistant that emits structured tool calls.

For deployment on Railway, each project has its own `requirements.txt` and `app.py`.

## Railway Branches

- `final-nlp` contains the NLP service at repository root.
- `final-speech` contains the Speech service at repository root.

Each service exposes `/health`, a browser UI at `/`, and API endpoints documented in the project README.
