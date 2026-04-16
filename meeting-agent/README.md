# Meeting Agent

Production pipeline: upload/URL → ffmpeg → (Demucs) → pyannote diarization → Resemblyzer speaker ID → Whisper (vLLM) → SQLite + Gradio.

Each model runs in its own container with its own pinned deps, so torch/torchaudio versions never collide.

## Prereqs
- Docker Desktop (WSL2 on Windows) with the NVIDIA Container Toolkit
- A GPU with enough VRAM. On 24 GB you can co-locate everything. On 16 GB, leave `VLLM_GPU_UTIL=0.5`. On 12 GB, consider swapping vLLM for faster-whisper.
- HF token with access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

## Setup
1. Copy `.env.example` to `.env` and fill in `HF_TOKEN`.
2. `docker compose up --build`
3. Open `http://localhost:7860`

Model downloads on first run take a few minutes and are cached in the `hf-cache` volume.

## Ports
- `7860` — Gradio UI + FastAPI (orchestrator)
- `8001` — vLLM OpenAI-compatible API
- `8002` — pyannote service
- `8003` — resemblyzer service

## Data layout
All services share `./data` mounted at `/data`:
- `data/jobs/<job_id>/` — per-job working files (audio, demucs output)
- `data/uploads/` — originals from UI uploads
- `data/enrollments/` — normalized enrollment clips
- `data/voice_profiles/` — pickled embeddings from resemblyzer
- `data/meeting_agent.db` — SQLite

## API (if you'd rather skip the UI)
- `POST /jobs` (multipart: `file` or `url`, flags: `denoise`, `identify_speakers`, `num_speakers`, `language`)
- `GET /jobs/{id}`
- `GET /jobs`
- `PATCH /jobs/{id}/speakers/{speaker_id}` (JSON: `{"display_name": "Alice"}`)
- `POST /voices/enroll` (multipart: `name`, `file`)
- `GET /voices`
