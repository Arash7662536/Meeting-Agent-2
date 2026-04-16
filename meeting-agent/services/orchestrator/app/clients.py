from pathlib import Path

import httpx

from .settings import settings


def pyannote_diarize(audio_path: str, num_speakers=None,
                     min_speakers=None, max_speakers=None) -> dict:
    payload = {"audio_path": audio_path}
    if num_speakers:
        payload["num_speakers"] = num_speakers
    if min_speakers:
        payload["min_speakers"] = min_speakers
    if max_speakers:
        payload["max_speakers"] = max_speakers
    r = httpx.post(f"{settings.PYANNOTE_URL}/diarize", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


def resemblyzer_identify(audio_path: str, segments: list[dict],
                         threshold: float = 0.75) -> dict:
    r = httpx.post(
        f"{settings.RESEMBLYZER_URL}/identify",
        json={"audio_path": audio_path, "segments": segments,
              "threshold": threshold},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def resemblyzer_enroll(name: str, audio_path: str) -> dict:
    r = httpx.post(
        f"{settings.RESEMBLYZER_URL}/enroll",
        json={"name": name, "audio_path": audio_path},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def resemblyzer_profiles() -> list[str]:
    r = httpx.get(f"{settings.RESEMBLYZER_URL}/profiles", timeout=30)
    r.raise_for_status()
    return r.json()["profiles"]


def resemblyzer_delete_profile(name: str) -> dict:
    r = httpx.delete(f"{settings.RESEMBLYZER_URL}/profiles/{name}", timeout=30)
    r.raise_for_status()
    return r.json()


def vllm_transcribe(audio_path: str, language: str | None = None) -> dict:
    path = Path(audio_path)
    with open(path, "rb") as f:
        files = {"file": (path.name, f.read(), "audio/wav")}
    data = {
        "model": settings.WHISPER_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    if language:
        data["language"] = language
    r = httpx.post(
        f"{settings.VLLM_URL}/v1/audio/transcriptions",
        files=files, data=data, timeout=3600,
    )
    r.raise_for_status()
    return r.json()
