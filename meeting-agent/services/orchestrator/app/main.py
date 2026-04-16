import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import clients
from .db import Job, Segment, SessionLocal, Speaker, init_db
from .settings import settings
from .tasks import enqueue_job
from .ui import build_ui

api = FastAPI(title="Meeting Agent")
_gradio_share = settings.GRADIO_SHARE


@api.on_event("startup")
def _startup() -> None:
    init_db()
    _launch_gradio()


def _launch_gradio() -> None:
    ui = build_ui()

    def _run():
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=_gradio_share,
            prevent_thread_lock=True,
            quiet=False,
        )

    threading.Thread(target=_run, daemon=True).start()


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    denoise: bool = Form(False),
    identify_speakers: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    language: Optional[str] = Form(None),
):
    if not file and not url:
        raise HTTPException(400, "provide file or url")

    if file:
        stage = settings.DATA_DIR / "uploads"
        stage.mkdir(parents=True, exist_ok=True)
        dest = stage / f"{uuid.uuid4().hex[:6]}_{file.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        jid = enqueue_job(
            str(dest), False, file.filename or dest.name,
            denoise, identify_speakers, num_speakers, language,
        )
    else:
        jid = enqueue_job(
            url, True, url,
            denoise, identify_speakers, num_speakers, language,
        )
    return {"job_id": jid}


@api.get("/jobs")
def list_jobs():
    with SessionLocal() as db:
        jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
        return [
            {
                "id": j.id, "status": j.status,
                "input_name": j.input_name,
                "created_at": j.created_at.isoformat(),
            }
            for j in jobs
        ]


@api.get("/jobs/{job_id}")
def get_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404)
        speakers = (
            db.query(Speaker).filter(Speaker.job_id == job_id).all()
        )
        segments = (
            db.query(Segment)
            .filter(Segment.job_id == job_id)
            .order_by(Segment.start)
            .all()
        )
        return {
            "id": job.id, "status": job.status, "progress": job.progress,
            "error": job.error, "num_speakers": job.num_speakers,
            "language": job.language,
            "speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "display_name": s.display_name,
                    "duration": s.total_duration,
                }
                for s in speakers
            ],
            "segments": [
                {"start": s.start, "end": s.end,
                 "speaker": s.speaker_id, "text": s.text}
                for s in segments
            ],
        }


class RenameSpeaker(BaseModel):
    display_name: str


@api.patch("/jobs/{job_id}/speakers/{speaker_id}")
def rename_speaker(job_id: str, speaker_id: str, body: RenameSpeaker):
    with SessionLocal() as db:
        spk = (
            db.query(Speaker)
            .filter(Speaker.job_id == job_id,
                    Speaker.speaker_id == speaker_id)
            .first()
        )
        if not spk:
            raise HTTPException(404)
        spk.display_name = body.display_name
        db.commit()
        return {"ok": True}


@api.post("/voices/enroll")
async def enroll_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    stage = settings.DATA_DIR / "enrollments"
    stage.mkdir(parents=True, exist_ok=True)
    tmp_path = stage / f"_tmp_{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    dest = stage / f"{name}_{uuid.uuid4().hex[:6]}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    tmp_path.unlink(missing_ok=True)

    rel = str(dest.relative_to(settings.DATA_DIR))
    return clients.resemblyzer_enroll(name, rel)


@api.get("/voices")
def list_voices():
    return {"profiles": clients.resemblyzer_profiles()}


@api.delete("/voices/{name}")
def delete_voice(name: str):
    return clients.resemblyzer_delete_profile(name)


app = api
