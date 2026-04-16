import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import clients
from .db import Job, Segment, SessionLocal, Speaker, init_db
from .settings import settings


def _run(cmd: list[str], capture: bool = True) -> None:
    subprocess.run(cmd, check=True, capture_output=capture, text=True)


def _extract_audio(input_path: Path, out_path: Path) -> Path:
    _run([
        "ffmpeg", "-y", "-i", str(input_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out_path),
    ])
    return out_path


def _download_url(url: str, out_dir: Path) -> Path:
    template = str(out_dir / "source.%(ext)s")
    _run(["yt-dlp", "-f", "bestaudio/best", "-o", template, url], capture=False)
    candidates = sorted(out_dir.glob("source.*"))
    if not candidates:
        raise RuntimeError("yt-dlp produced no output file")
    return candidates[0]


def _denoise_demucs(audio_path: Path, work_dir: Path) -> Path:
    out_root = work_dir / "demucs_out"
    _run([
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals", "-n", "htdemucs",
        "-o", str(out_root), str(audio_path),
    ], capture=False)
    vocals = out_root / "htdemucs" / audio_path.stem / "vocals.wav"
    if not vocals.exists():
        raise RuntimeError(f"demucs output missing: {vocals}")
    cleaned = work_dir / f"{audio_path.stem}_clean.wav"
    _run([
        "ffmpeg", "-y", "-i", str(vocals),
        "-ar", "16000", "-ac", "1", str(cleaned),
    ])
    return cleaned


def _assign_speaker(seg_start: float, seg_end: float, turns: list[dict]) -> str:
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(seg_end, t["end"]) - max(seg_start, t["start"])
        if overlap > best_overlap:
            best_overlap, best = overlap, t["speaker"]
    return best or "UNKNOWN"


def _set_status(job_id: str, status: str, progress: str = "") -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = status
        job.progress = progress
        db.commit()


def run_pipeline(
    job_id: str,
    input_path: str,
    is_url: bool,
    denoise: bool = False,
    identify_speakers: bool = True,
    num_speakers: Optional[int] = None,
    language: Optional[str] = None,
) -> None:
    init_db()
    job_dir = settings.DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        _set_status(job_id, "extracting", "fetching source")
        src = _download_url(input_path, job_dir) if is_url else Path(input_path)

        _set_status(job_id, "extracting", "decoding audio")
        audio = _extract_audio(src, job_dir / "audio_16k.wav")

        if denoise:
            _set_status(job_id, "denoising", "running Demucs")
            audio = _denoise_demucs(audio, job_dir)

        rel_audio = str(audio.relative_to(settings.DATA_DIR))
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job:
                job.audio_path = rel_audio
                db.commit()

        _set_status(job_id, "diarizing", "pyannote")
        diar = clients.pyannote_diarize(rel_audio, num_speakers=num_speakers)
        turns: list[dict] = diar["turns"]

        speaker_map: dict[str, str] = {}
        if identify_speakers:
            try:
                _set_status(job_id, "identifying", "resemblyzer")
                res = clients.resemblyzer_identify(rel_audio, turns)
                speaker_map = {k: v for k, v in res["mapping"].items() if v}
            except Exception as e:
                print(f"speaker identification skipped: {e}")

        _set_status(job_id, "transcribing", "whisper via vLLM")
        transcription = clients.vllm_transcribe(str(audio), language=language)
        whisper_segments = transcription.get("segments") or []

        speaker_totals: dict[str, float] = defaultdict(float)
        for t in turns:
            speaker_totals[t["speaker"]] += t["end"] - t["start"]

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.num_speakers = diar["num_speakers"]
            if language:
                job.language = language
            elif transcription.get("language"):
                job.language = transcription["language"]

            for spk, dur in speaker_totals.items():
                db.add(Speaker(
                    job_id=job_id, speaker_id=spk,
                    display_name=speaker_map.get(spk, spk),
                    total_duration=dur,
                ))

            for ws in whisper_segments:
                start = float(ws.get("start", 0.0))
                end = float(ws.get("end", start))
                text = (ws.get("text") or "").strip()
                spk = _assign_speaker(start, end, turns)
                db.add(Segment(
                    job_id=job_id, start=start, end=end,
                    speaker_id=spk, text=text,
                ))

            job.status = "completed"
            job.progress = ""
            job.completed_at = datetime.utcnow()
            db.commit()

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error = f"{e}\n{tb}"
                db.commit()
        raise
