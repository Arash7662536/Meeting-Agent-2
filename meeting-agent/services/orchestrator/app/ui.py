import shutil
import subprocess
import uuid
from pathlib import Path

import gradio as gr

from . import clients
from .db import Job, Segment, SessionLocal, Speaker
from .settings import settings
from .tasks import enqueue_job


def _submit(file, url, denoise, identify, num_speakers, language):
    if not file and not (url and url.strip()):
        return "provide a file or a URL"

    num = int(num_speakers) if num_speakers else None
    lang = (language or "").strip() or None

    if file:
        stage = settings.DATA_DIR / "uploads"
        stage.mkdir(parents=True, exist_ok=True)
        dest = stage / f"{uuid.uuid4().hex[:6]}_{Path(file.name).name}"
        shutil.copy(file.name, dest)
        return enqueue_job(
            str(dest), False, dest.name,
            bool(denoise), bool(identify), num, lang,
        )

    u = url.strip()
    return enqueue_job(u, True, u, bool(denoise), bool(identify), num, lang)


def _fetch(job_id):
    if not job_id:
        return "", [], []
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return "not found", [], []
        speakers = (
            db.query(Speaker).filter(Speaker.job_id == job_id).all()
        )
        segments = (
            db.query(Segment)
            .filter(Segment.job_id == job_id)
            .order_by(Segment.start)
            .all()
        )

    parts = [job.status]
    if job.progress:
        parts.append(job.progress)
    status = " — ".join(parts)
    if job.error:
        status += f"\n\nERROR: {job.error.splitlines()[0]}"

    spk_rows = [
        [s.speaker_id, s.display_name, f"{s.total_duration:.1f}s"]
        for s in speakers
    ]
    seg_rows = [
        [f"{s.start:.2f}", f"{s.end:.2f}", s.speaker_id, s.text]
        for s in segments
    ]
    return status, spk_rows, seg_rows


def _rename(job_id, speaker_id, new_name):
    if job_id and speaker_id and new_name:
        with SessionLocal() as db:
            spk = (
                db.query(Speaker)
                .filter(Speaker.job_id == job_id,
                        Speaker.speaker_id == speaker_id)
                .first()
            )
            if spk:
                spk.display_name = new_name
                db.commit()
    return _fetch(job_id)


def _list_jobs():
    with SessionLocal() as db:
        jobs = (
            db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
        )
        return [
            [j.id, j.status, j.input_name, j.created_at.isoformat(timespec="seconds")]
            for j in jobs
        ]


def _list_profiles():
    try:
        return {"profiles": clients.resemblyzer_profiles()}
    except Exception as e:
        return {"error": str(e)}


def _enroll(name, f):
    if not name or not f:
        return "need name + file"
    stage = settings.DATA_DIR / "enrollments"
    stage.mkdir(parents=True, exist_ok=True)
    dest = stage / f"{name}_{uuid.uuid4().hex[:6]}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", f.name, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    rel = str(dest.relative_to(settings.DATA_DIR))
    try:
        return clients.resemblyzer_enroll(name, rel)
    except Exception as e:
        return {"error": str(e)}


def _delete_profile(name):
    if not name:
        return _list_profiles()
    try:
        clients.resemblyzer_delete_profile(name)
    except Exception as e:
        return {"error": str(e)}
    return _list_profiles()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Meeting Agent") as ui:
        gr.Markdown("# Meeting Agent")

        with gr.Tab("Transcribe"):
            with gr.Row():
                file_in = gr.File(
                    label="Audio/Video file",
                    file_types=["audio", "video"],
                )
                url_in = gr.Textbox(label="Or paste URL (yt-dlp supported)")
            with gr.Row():
                denoise = gr.Checkbox(label="Denoise (Demucs)", value=False)
                identify = gr.Checkbox(label="Identify speakers", value=True)
                num_spk = gr.Number(
                    label="Num speakers (optional)", precision=0
                )
                lang = gr.Textbox(label="Language (optional, e.g. 'en')")
            submit = gr.Button("Submit", variant="primary")
            job_id_box = gr.Textbox(label="Job ID (copy this)")
            submit.click(
                _submit,
                [file_in, url_in, denoise, identify, num_spk, lang],
                job_id_box,
            )

        with gr.Tab("Results"):
            jid_in = gr.Textbox(label="Job ID")
            refresh = gr.Button("Refresh")
            status_out = gr.Textbox(label="Status", lines=2)
            speakers_tbl = gr.Dataframe(
                headers=["speaker_id", "display_name", "duration"],
                interactive=False, label="Speakers",
            )
            segments_tbl = gr.Dataframe(
                headers=["start", "end", "speaker", "text"],
                interactive=False, label="Transcript", wrap=True,
            )
            refresh.click(
                _fetch, jid_in, [status_out, speakers_tbl, segments_tbl]
            )

            gr.Markdown("### Rename speaker")
            with gr.Row():
                spk_id = gr.Textbox(label="Speaker ID (e.g. SPEAKER_00)")
                new_name = gr.Textbox(label="New name")
                rename_btn = gr.Button("Rename")
            rename_btn.click(
                _rename, [jid_in, spk_id, new_name],
                [status_out, speakers_tbl, segments_tbl],
            )

        with gr.Tab("Jobs"):
            jobs_tbl = gr.Dataframe(
                headers=["id", "status", "input", "created"],
                interactive=False,
            )
            gr.Button("Refresh").click(_list_jobs, None, jobs_tbl)

        with gr.Tab("Voice profiles"):
            prof_list = gr.JSON(label="Enrolled")
            gr.Button("Refresh").click(_list_profiles, None, prof_list)
            gr.Markdown("### Enroll a new voice (clean ~10s clip)")
            name_in = gr.Textbox(label="Name")
            voice_file = gr.File(
                label="Audio", file_types=["audio", "video"]
            )
            enroll_status = gr.JSON()
            gr.Button("Enroll").click(
                _enroll, [name_in, voice_file], enroll_status
            )
            gr.Markdown("### Delete")
            del_name = gr.Textbox(label="Name to delete")
            gr.Button("Delete").click(_delete_profile, del_name, prof_list)

    return ui
