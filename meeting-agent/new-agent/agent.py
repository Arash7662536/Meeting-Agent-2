#!/usr/bin/env python3
"""
Persian Meeting Agent
──────────────────────────────────────────────────────────────────────────────
Transcription : Whisper via vLLM  (default → http://localhost:8001)
Diarization   : DiariZen Large v2  (default)
              | pyannote community-1
              | pyannote 3.1 (legacy)

Supported inputs
  • Local audio file  : mp3, wav, flac, ogg, m4a …
  • Local video file  : mp4, mkv, avi, mov …
  • YouTube URL       : https://youtube.com/watch?v=…  or  youtu.be/…
  • Any direct URL    : https://example.com/recording.mp4

Usage
──────────────────────────────────────────────────────────────────────────────
  python agent.py meeting.mp4
  python agent.py https://youtu.be/XXXX --model community-1
  python agent.py meeting.wav --out-srt out.srt --out-json out.json
  python agent.py meeting.mp4 --model pyannote-3.1 --language fa
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import urlparse

import requests
import torch

# ── Logging ────────────────────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _GREEN  = Fore.GREEN
    _YELLOW = Fore.YELLOW
    _CYAN   = Fore.CYAN
    _RESET  = Style.RESET_ALL
    _BOLD   = Style.BRIGHT
except ImportError:
    _GREEN = _YELLOW = _CYAN = _RESET = _BOLD = ""

logging.basicConfig(
    level=logging.INFO,
    format=f"{_CYAN}%(asctime)s{_RESET} [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meeting-agent")

# ── Types ──────────────────────────────────────────────────────────────────────

DiarizationModel = Literal["diarizen", "community-1", "pyannote-3.1"]

MODEL_IDS: dict[DiarizationModel, str] = {
    "diarizen":    "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    "community-1": "pyannote/speaker-diarization-community-1",
    "pyannote-3.1":"pyannote/speaker-diarization-3.1",
}

@dataclass
class Segment:
    start:   float
    end:     float
    speaker: str
    text:    str = ""

    def duration(self) -> float:
        return round(self.end - self.start, 3)


# ── Audio extraction ───────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


def extract_audio(source: str, output_dir: str) -> str:
    """
    Accept any source → return path to a 16 kHz mono WAV file.

    Routing:
      YouTube URL  → yt-dlp  → ffmpeg resample
      Other URL    → requests download → ffmpeg
      Video file   → ffmpeg (strip video, resample)
      Audio file   → ffmpeg (resample only if needed)
    """
    out_wav = os.path.join(output_dir, "audio.wav")
    parsed  = urlparse(source)
    is_url  = parsed.scheme in ("http", "https")

    if is_url and _is_youtube(parsed.netloc):
        _download_youtube(source, out_wav, output_dir)

    elif is_url:
        tmp = os.path.join(output_dir, "download" + Path(parsed.path).suffix or ".bin")
        log.info("Downloading from URL …")
        _http_download(source, tmp)
        _ffmpeg_to_wav16k(tmp, out_wav)

    else:
        suffix = Path(source).suffix.lower()
        if suffix not in VIDEO_EXTS | AUDIO_EXTS:
            log.warning(f"Unknown extension '{suffix}' — trying ffmpeg anyway")
        log.info(f"Processing local file: {source}")
        _ffmpeg_to_wav16k(source, out_wav)

    if not os.path.exists(out_wav):
        raise FileNotFoundError(f"Audio extraction failed — {out_wav} not produced")

    log.info(f"Audio ready → {out_wav}  ({_file_mb(out_wav):.1f} MB)")
    return out_wav


def _is_youtube(netloc: str) -> bool:
    return any(h in netloc for h in ("youtube.com", "youtu.be", "www.youtube.com"))


def _download_youtube(url: str, out_wav: str, output_dir: str):
    log.info("Downloading YouTube audio via yt-dlp …")
    tmpl = os.path.join(output_dir, "yt_audio.%(ext)s")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", tmpl,
                url,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Install with:  pip install yt-dlp")

    # yt-dlp produces yt_audio.wav; resample to 16 kHz mono
    candidates = sorted(Path(output_dir).glob("yt_audio.*"))
    if not candidates:
        raise RuntimeError("yt-dlp produced no output file")
    _ffmpeg_to_wav16k(str(candidates[0]), out_wav)


def _http_download(url: str, dest: str):
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65_536):
            f.write(chunk)
    log.info(f"Downloaded → {dest}  ({_file_mb(dest):.1f} MB)")


def _ffmpeg_to_wav16k(src: str, dst: str):
    """Convert any audio/video to 16 kHz mono PCM WAV."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-ar", "16000",   # 16 kHz
                "-ac", "1",        # mono
                "-vn",             # drop video
                "-f", "wav",
                dst,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with:  apt install ffmpeg  or  brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr.decode()}")


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


# ── Diarization ────────────────────────────────────────────────────────────────

class Diarizer:
    """
    Unified wrapper for all three diarization backends.

    All three expose the same pyannote-style pipeline interface:
        pipeline({"waveform": ..., "sample_rate": ...})

    DiariZen requires extra install:
        pip install git+https://github.com/BUTSpeechFIT/DiariZen
    """

    def __init__(
        self,
        model: DiarizationModel = "diarizen",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model
        self.hf_token   = hf_token or os.environ.get("HF_TOKEN", "")
        self.device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._pipeline  = None

        log.info(f"Loading diarization model:  {_BOLD}{model}{_RESET}  on {self.device}")
        self._load()
        log.info(f"{_GREEN}Diarization model loaded ✓{_RESET}")

    # ── loading ────────────────────────────────────────────────────────────────

    def _load(self):
        if self.model_name in ("pyannote-3.1", "community-1"):
            self._load_pyannote(MODEL_IDS[self.model_name])
        elif self.model_name == "diarizen":
            self._load_diarizen()
        else:
            raise ValueError(f"Unknown diarization model: {self.model_name}")

    def _load_pyannote(self, model_id: str):
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("Install pyannote:  pip install pyannote.audio")

        self._pipeline = Pipeline.from_pretrained(
            model_id,
            use_auth_token=self.hf_token if self.hf_token else None,
        )
        self._pipeline.to(self.device)

    def _load_diarizen(self):
        """
        DiariZen is built on pyannote infrastructure but uses a custom
        WavLM-based segmentation model.  Requires the DiariZen package:

            pip install git+https://github.com/BUTSpeechFIT/DiariZen

        The model weights are on HuggingFace (CC BY-NC 4.0 — non-commercial):
            BUT-FIT/diarizen-wavlm-large-s80-md-v2
        """
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                MODEL_IDS["diarizen"],
                use_auth_token=self.hf_token if self.hf_token else None,
            )
            self._pipeline.to(self.device)

        except Exception as exc:
            # Provide a clear, actionable error if DiariZen isn't installed
            msg = str(exc)
            if "DiariZen" in msg or "diarizen" in msg.lower() or "BUT-FIT" in msg:
                raise RuntimeError(
                    "\n\nDiariZen model failed to load.\n"
                    "Make sure you have installed the DiariZen package:\n\n"
                    "    pip install git+https://github.com/BUTSpeechFIT/DiariZen\n\n"
                    "Then accept the model conditions on HuggingFace:\n"
                    "    https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2\n\n"
                    f"Original error: {exc}"
                ) from exc
            raise

    # ── inference ──────────────────────────────────────────────────────────────

    def diarize(self, audio_path: str) -> List[Segment]:
        """
        Run diarization on a WAV file.
        Returns a time-sorted list of Segment(start, end, speaker).
        """
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        log.info("Running diarization … (this may take a moment)")

        annotation = self._pipeline({"waveform": waveform, "sample_rate": sr})

        segments: List[Segment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(Segment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=speaker,
            ))

        n_speakers = len({s.speaker for s in segments})
        log.info(
            f"{_GREEN}Diarization done ✓{_RESET}  "
            f"→  {n_speakers} speaker(s), {len(segments)} segment(s)"
        )
        return segments


# ── Transcription ──────────────────────────────────────────────────────────────

class Transcriber:
    """
    Sends audio to a Whisper model served by vLLM via the OpenAI-compatible
    /v1/audio/transcriptions  endpoint.

    The vLLM server should be started with something like:
        vllm serve openai/whisper-large-v3 --port 8001
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model:    str = "whisper-large-v3",
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.endpoint = f"{self.base_url}/v1/audio/transcriptions"

    def transcribe(self, audio_path: str, language: str = "fa") -> dict:
        """
        POST the audio file to vLLM Whisper and return the verbose_json response.

        The response contains:
            {
              "text": "...",
              "segments": [
                { "id": 0, "start": 0.0, "end": 2.5, "text": "...", ... },
                ...
              ]
            }
        """
        log.info(f"Transcribing via Whisper at {self.endpoint}  (language={language}) …")

        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    self.endpoint,
                    files={"file": (Path(audio_path).name, f, "audio/wav")},
                    data={
                        "model":              self.model,
                        "language":           language,
                        "response_format":    "verbose_json",
                        # request segment-level timestamps for alignment
                        "timestamp_granularities[]": "segment",
                    },
                    timeout=600,   # long audio can legitimately take minutes
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"\nCannot reach Whisper at {self.endpoint}\n"
                "Make sure vLLM is running, e.g.:\n"
                f"    vllm serve openai/whisper-large-v3 --port 8001\n"
            )

        if not resp.ok:
            raise RuntimeError(
                f"Whisper returned HTTP {resp.status_code}:\n{resp.text[:500]}"
            )

        result = resp.json()
        n_segs = len(result.get("segments", []))
        log.info(f"{_GREEN}Transcription done ✓{_RESET}  →  {n_segs} segment(s)")
        return result


# ── Alignment ──────────────────────────────────────────────────────────────────

def align_speaker_to_transcript(
    diarization:   List[Segment],
    transcription: dict,
) -> List[Segment]:
    """
    Match Whisper transcript segments → diarization speaker labels.

    Strategy: for each Whisper segment [ws, we], find the diarization
    speaker whose time intervals have the greatest total overlap with [ws, we].
    Falls back to "Unknown" if no overlap is found.
    """
    whisper_segments = transcription.get("segments", [])
    aligned: List[Segment] = []

    for wseg in whisper_segments:
        ws   = float(wseg["start"])
        we   = float(wseg["end"])
        text = wseg.get("text", "").strip()

        if not text:
            continue

        # accumulate overlap per speaker
        overlap_per_speaker: dict[str, float] = {}
        for dseg in diarization:
            overlap = max(0.0, min(we, dseg.end) - max(ws, dseg.start))
            if overlap > 0:
                overlap_per_speaker[dseg.speaker] = (
                    overlap_per_speaker.get(dseg.speaker, 0.0) + overlap
                )

        best_speaker = (
            max(overlap_per_speaker, key=overlap_per_speaker.get)
            if overlap_per_speaker
            else "Unknown"
        )

        aligned.append(Segment(start=ws, end=we, speaker=best_speaker, text=text))

    return aligned


# ── Output ─────────────────────────────────────────────────────────────────────

def merge_consecutive(segments: List[Segment], gap_threshold: float = 1.5) -> List[Segment]:
    """
    Merge adjacent segments from the same speaker with a gap ≤ gap_threshold seconds.
    Keeps the transcript readable without excessive fragmentation.
    """
    if not segments:
        return []

    merged = [Segment(
        start=segments[0].start,
        end=segments[0].end,
        speaker=segments[0].speaker,
        text=segments[0].text,
    )]

    for seg in segments[1:]:
        last = merged[-1]
        same_speaker = seg.speaker == last.speaker
        small_gap    = (seg.start - last.end) <= gap_threshold

        if same_speaker and small_gap:
            last.end   = seg.end
            last.text += " " + seg.text
        else:
            merged.append(Segment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=seg.text,
            ))

    return merged


def format_transcript(segments: List[Segment]) -> str:
    """Human-readable transcript with speaker labels and timestamps."""
    return "\n".join(
        f"[{_fmt_time(s.start)}]  {_BOLD}{s.speaker}{_RESET}: {s.text}"
        for s in segments
    )


def export_txt(segments: List[Segment], path: str):
    txt = "\n".join(f"[{_fmt_time(s.start)}]  {s.speaker}: {s.text}" for s in segments)
    Path(path).write_text(txt, encoding="utf-8")
    log.info(f"Saved TXT  → {path}")


def export_json(segments: List[Segment], path: str):
    data = [
        {
            "start":    s.start,
            "end":      s.end,
            "duration": s.duration(),
            "speaker":  s.speaker,
            "text":     s.text,
        }
        for s in segments
    ]
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"Saved JSON → {path}")


def export_srt(segments: List[Segment], path: str):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.extend([
            str(i),
            f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}",
            f"{seg.speaker}: {seg.text}",
            "",
        ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved SRT  → {path}")


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_ts(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    ms  = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(
    source:            str,
    diarization_model: DiarizationModel = "diarizen",
    whisper_url:       str = "http://localhost:8001",
    whisper_model:     str = "whisper-large-v3",
    language:          str = "fa",
    hf_token:          Optional[str] = None,
    out_txt:           Optional[str] = None,
    out_json:          Optional[str] = None,
    out_srt:           Optional[str] = None,
    device:            Optional[str] = None,
    merge_gap:         float = 1.5,
) -> List[Segment]:
    """
    Full pipeline:
        1. Extract audio  (ffmpeg / yt-dlp)
        2. Diarize        (chosen model)
        3. Transcribe     (Whisper @ vLLM)
        4. Align          (speaker × transcript segments)
        5. Output         (print + optional files)
    """
    print(f"\n{_BOLD}{'─'*60}")
    print(f"  Persian Meeting Agent")
    print(f"  Diarization : {diarization_model}")
    print(f"  Whisper     : {whisper_url}  [{whisper_model}]")
    print(f"  Language    : {language}")
    print(f"{'─'*60}{_RESET}\n")

    with tempfile.TemporaryDirectory(prefix="meeting_agent_") as tmpdir:
        # ── 1. Audio ────────────────────────────────────────────────────────────
        audio_path = extract_audio(source, tmpdir)

        # ── 2. Diarize ──────────────────────────────────────────────────────────
        diarizer      = Diarizer(model=diarization_model, hf_token=hf_token, device=device)
        diar_segments = diarizer.diarize(audio_path)

        # ── 3. Transcribe ───────────────────────────────────────────────────────
        transcriber   = Transcriber(base_url=whisper_url, model=whisper_model)
        transcription = transcriber.transcribe(audio_path, language=language)

        # ── 4. Align ────────────────────────────────────────────────────────────
        segments = align_speaker_to_transcript(diar_segments, transcription)
        segments = merge_consecutive(segments, gap_threshold=merge_gap)

        # ── 5. Output ───────────────────────────────────────────────────────────
        print(f"\n{_BOLD}{'═'*60}")
        print("  TRANSCRIPT")
        print(f"{'═'*60}{_RESET}\n")
        print(format_transcript(segments))
        print(f"\n{_BOLD}{'═'*60}{_RESET}\n")

        speakers = {s.speaker for s in segments}
        print(f"  Speakers detected : {len(speakers)}  →  {', '.join(sorted(speakers))}")
        print(f"  Total segments    : {len(segments)}")
        print()

        if out_txt:
            export_txt(segments, out_txt)
        if out_json:
            export_json(segments, out_json)
        if out_srt:
            export_srt(segments, out_srt)

    return segments


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent",
        description="Persian Meeting Agent — speaker diarization + Whisper transcription",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python agent.py meeting.mp4
  python agent.py https://youtu.be/XXXXX --model community-1
  python agent.py recording.wav --out-srt meeting.srt --out-json meeting.json
  python agent.py meeting.mp4 --model pyannote-3.1 --language fa

Diarization models (--model):
  diarizen      DiariZen Large v2 — best accuracy   [DEFAULT]
  community-1   pyannote community-1 — great balance
  pyannote-3.1  pyannote 3.1 legacy — widest compat

HuggingFace token:
  Required for all three models (they are gated).
  Set via  --hf-token  or  export HF_TOKEN=hf_...
        """,
    )

    p.add_argument("source", help="File path, YouTube URL, or any media URL")

    p.add_argument(
        "--model", "-m",
        choices=["diarizen", "community-1", "pyannote-3.1"],
        default="diarizen",
        metavar="MODEL",
        help="Diarization model  (default: diarizen)",
    )

    p.add_argument(
        "--whisper-url",
        default="http://localhost:8001",
        metavar="URL",
        help="vLLM Whisper base URL  (default: http://localhost:8001)",
    )
    p.add_argument(
        "--whisper-model",
        default="whisper-large-v3",
        metavar="NAME",
        help="Whisper model name as served by vLLM  (default: whisper-large-v3)",
    )
    p.add_argument(
        "--language", "-l",
        default="fa",
        metavar="LANG",
        help="BCP-47 language code for transcription  (default: fa = Persian)",
    )

    p.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace access token  (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="Torch device for diarization: cuda / cpu  (auto-detected by default)",
    )

    p.add_argument("--out-txt",  default=None, metavar="FILE", help="Save plain-text transcript")
    p.add_argument("--out-json", default=None, metavar="FILE", help="Save JSON transcript")
    p.add_argument("--out-srt",  default=None, metavar="FILE", help="Save SRT subtitles")

    p.add_argument(
        "--merge-gap",
        type=float,
        default=1.5,
        metavar="SECS",
        help="Max gap (seconds) between same-speaker segments to merge  (default: 1.5)",
    )

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    try:
        run(
            source            = args.source,
            diarization_model = args.model,
            whisper_url       = args.whisper_url,
            whisper_model     = args.whisper_model,
            language          = args.language,
            hf_token          = args.hf_token,
            out_txt           = args.out_txt,
            out_json          = args.out_json,
            out_srt           = args.out_srt,
            device            = args.device,
            merge_gap         = args.merge_gap,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        log.error(f"{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
