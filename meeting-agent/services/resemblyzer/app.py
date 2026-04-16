import os
import pickle
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PROFILES_DIR = DATA_DIR / "voice_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from resemblyzer import VoiceEncoder
    state["encoder"] = VoiceEncoder()
    print("resemblyzer ready")
    yield
    state.clear()


app = FastAPI(title="resemblyzer service", lifespan=lifespan)


class EnrollRequest(BaseModel):
    name: str
    audio_path: str


class Segment(BaseModel):
    start: float
    end: float
    speaker: str


class IdentifyRequest(BaseModel):
    audio_path: str
    segments: list[Segment]
    threshold: float = 0.75


class IdentifyResponse(BaseModel):
    mapping: dict[str, Optional[str]]
    scores: dict[str, float]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else DATA_DIR / path


def _embed_file(path: Path) -> np.ndarray:
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(path)
    return state["encoder"].embed_utterance(wav)


def _embed_segment(wav: np.ndarray, start: float, end: float, sr: int = 16000):
    s, e = int(start * sr), int(end * sr)
    if e - s < int(sr * 0.5):
        return None
    return state["encoder"].embed_utterance(wav[s:e])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/profiles")
def list_profiles():
    return {"profiles": [p.stem for p in PROFILES_DIR.glob("*.pkl")]}


@app.post("/enroll")
def enroll(req: EnrollRequest):
    path = _resolve(req.audio_path)
    if not path.exists():
        raise HTTPException(404, f"audio not found: {path}")
    emb = _embed_file(path)
    out = PROFILES_DIR / f"{req.name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(emb, f)
    return {"name": req.name, "embedding_dim": len(emb)}


@app.delete("/profiles/{name}")
def delete_profile(name: str):
    out = PROFILES_DIR / f"{name}.pkl"
    existed = out.exists()
    if existed:
        out.unlink()
    return {"deleted": name, "existed": existed}


@app.post("/identify", response_model=IdentifyResponse)
def identify(req: IdentifyRequest):
    from resemblyzer import preprocess_wav

    path = _resolve(req.audio_path)
    if not path.exists():
        raise HTTPException(404, f"audio not found: {path}")

    profiles = {}
    for p in PROFILES_DIR.glob("*.pkl"):
        with open(p, "rb") as f:
            profiles[p.stem] = pickle.load(f)

    if not profiles:
        return IdentifyResponse(
            mapping={s.speaker: None for s in req.segments},
            scores={},
        )

    wav = preprocess_wav(path)

    by_speaker: dict[str, list[Segment]] = defaultdict(list)
    for seg in req.segments:
        by_speaker[seg.speaker].append(seg)

    mapping: dict[str, Optional[str]] = {}
    scores: dict[str, float] = {}

    for speaker_id, segs in by_speaker.items():
        embs = []
        for seg in segs:
            emb = _embed_segment(wav, seg.start, seg.end)
            if emb is not None:
                embs.append(emb)
        if not embs:
            mapping[speaker_id] = None
            continue
        mean_emb = np.mean(embs, axis=0)
        best_name, best_score = None, -1.0
        for name, prof_emb in profiles.items():
            sim = float(np.inner(mean_emb, prof_emb))
            if sim > best_score:
                best_score, best_name = sim, name
        scores[speaker_id] = best_score
        mapping[speaker_id] = best_name if best_score >= req.threshold else None

    return IdentifyResponse(mapping=mapping, scores=scores)
