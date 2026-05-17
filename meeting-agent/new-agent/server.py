#!/usr/bin/env python3
"""
Persian Meeting Agent — FastAPI backend
Serves the UI at http://localhost:7860
"""

from __future__ import annotations

import os
import json
import uuid
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import (
    extract_audio,
    Diarizer,
    Transcriber,
    align_speaker_to_transcript,
    merge_consecutive,
    export_txt,
    export_json,
    export_srt,
    DiarizationModel,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Persian Meeting Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

# In-memory job store  {job_id: job_dict}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── Job helpers ────────────────────────────────────────────────────────────────

def _new_job(job_id: str) -> Path:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    with _jobs_lock:
        _jobs[job_id] = {
            "id":         job_id,
            "status":     "pending",   # pending | running | done | error
            "step":       "",          # extracting | diarizing | transcribing | aligning | done
            "progress":   0,           # 0–100
            "message":    "Queued…",
            "error":      None,
            "result":     None,        # populated on success
            "created_at": datetime.now().isoformat(),
        }
    return job_dir


def _update(job_id: str, **kw):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)


def _get(job_id: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


# ── Background worker ──────────────────────────────────────────────────────────

def _run_pipeline(
    job_id:       str,
    source:       str,
    model:        DiarizationModel,
    whisper_url:  str,
    whisper_model:str,
    language:     str,
    hf_token:     Optional[str],
    merge_gap:    float,
):
    job_dir = JOBS_DIR / job_id
    try:
        # 1 ── extract audio
        _update(job_id, status="running", step="extracting", progress=5,
                message="Extracting audio from source…")
        audio_dir = str(job_dir / "audio")
        Path(audio_dir).mkdir(exist_ok=True)
        audio_path = extract_audio(source, audio_dir)

        # 2 ── diarize
        _update(job_id, step="diarizing", progress=20,
                message=f"Running diarization  [{model}]…")
        diarizer      = Diarizer(model=model, hf_token=hf_token)
        diar_segments = diarizer.diarize(audio_path)

        # 3 ── transcribe
        _update(job_id, step="transcribing", progress=60,
                message="Sending audio to Whisper…")
        transcriber   = Transcriber(base_url=whisper_url, model=whisper_model)
        transcription = transcriber.transcribe(audio_path, language=language)

        # 4 ── align
        _update(job_id, step="aligning", progress=85,
                message="Aligning speakers to transcript…")
        segments = align_speaker_to_transcript(diar_segments, transcription)
        segments = merge_consecutive(segments, gap_threshold=merge_gap)

        # 5 ── export files
        _update(job_id, step="exporting", progress=92, message="Saving output files…")
        txt_path  = str(job_dir / "transcript.txt")
        json_path = str(job_dir / "transcript.json")
        srt_path  = str(job_dir / "transcript.srt")
        export_txt(segments, txt_path)
        export_json(segments, json_path)
        export_srt(segments, srt_path)

        # 6 ── done
        speakers = sorted({s.speaker for s in segments})
        total_duration = segments[-1].end if segments else 0

        _update(
            job_id,
            status="done", step="done", progress=100, message="Complete",
            result={
                "n_speakers": len(speakers),
                "speakers":   speakers,
                "n_segments": len(segments),
                "duration":   round(total_duration, 1),
                "segments": [
                    {
                        "start":   s.start,
                        "end":     s.end,
                        "speaker": s.speaker,
                        "text":    s.text,
                    }
                    for s in segments
                ],
            },
            # store file paths server-side only
            _files={"txt": txt_path, "json": json_path, "srt": srt_path},
        )

    except Exception as exc:
        import traceback
        _update(job_id, status="error", step="error", progress=0,
                message="Processing failed", error=str(exc))
        print(traceback.format_exc())


# ── API ────────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def start_process(
    file:          Optional[UploadFile] = File(None),
    url:           Optional[str]        = Form(None),
    model:         str                  = Form("diarizen"),
    whisper_url:   str                  = Form("http://localhost:8001"),
    whisper_model: str                  = Form("whisper-large-v3"),
    language:      str                  = Form("fa"),
    hf_token:      Optional[str]        = Form(None),
    merge_gap:     float                = Form(1.5),
):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide a file or a URL.")

    job_id  = str(uuid.uuid4())[:8]
    job_dir = _new_job(job_id)

    if file:
        save_path = str(job_dir / file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source = save_path
    else:
        source = url.strip()

    effective_token = hf_token or os.environ.get("HF_TOKEN") or None

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, source, model, whisper_url, whisper_model,
              language, effective_token, merge_gap),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # strip internal file paths — client should never see local FS paths
    job.pop("_files", None)
    return job


@app.get("/api/download/{job_id}/{fmt}")
def download_file(job_id: str, fmt: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job not finished")
    if fmt not in ("txt", "json", "srt"):
        raise HTTPException(status_code=400, detail="Unknown format")

    files = job.get("_files", {})
    path  = files.get(fmt)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    media = {"txt": "text/plain", "json": "application/json", "srt": "text/plain"}
    return FileResponse(path, filename=f"meeting_{job_id}.{fmt}",
                        media_type=media[fmt])


# ── Serve frontend (must be last) ─────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    print("\n  Meeting Agent UI  →  http://localhost:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
