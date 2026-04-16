import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pyannote.audio import Pipeline

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )

    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            pipeline.to(torch.device("cuda"))
            device = "cuda"
        except RuntimeError as e:
            print(f"GPU init failed, using CPU: {e}")

    state["pipeline"] = pipeline
    state["device"] = device
    print(f"pyannote ready on {device}")
    yield
    state.clear()


app = FastAPI(title="pyannote service", lifespan=lifespan)


class DiarizeRequest(BaseModel):
    audio_path: str
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class Turn(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    turns: list[Turn]
    num_speakers: int
    device: str


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else DATA_DIR / path


def _load_audio(path: Path):
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T)
    if sr != 16000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sr}


def _iter_turns(diarization):
    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            yield turn.start, turn.end, speaker
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield turn.start, turn.end, speaker
    else:
        raise RuntimeError(f"unknown diarization output: {type(diarization)}")


@app.get("/health")
def health():
    return {"status": "ok", "device": state.get("device")}


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    path = _resolve(req.audio_path)
    if not path.exists():
        raise HTTPException(404, f"audio_path not found: {path}")

    kwargs = {}
    if req.num_speakers:
        kwargs["num_speakers"] = req.num_speakers
    else:
        if req.min_speakers:
            kwargs["min_speakers"] = req.min_speakers
        if req.max_speakers:
            kwargs["max_speakers"] = req.max_speakers

    audio = _load_audio(path)
    diarization = state["pipeline"](audio, **kwargs)

    turns = [
        Turn(start=s, end=e, speaker=spk)
        for s, e, spk in _iter_turns(diarization)
    ]
    speakers = {t.speaker for t in turns}
    return DiarizeResponse(
        turns=turns, num_speakers=len(speakers), device=state["device"]
    )
