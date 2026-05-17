# Persian Meeting Agent — Setup Guide

## System requirements

```bash
# ffmpeg  (audio/video extraction)
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg              # macOS

# yt-dlp  (YouTube download) — also installed via pip
pip install yt-dlp
```

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## DiariZen (default model — extra step)

DiariZen is NOT on PyPI. Install it directly from GitHub:

```bash
pip install git+https://github.com/BUTSpeechFIT/DiariZen
```

Then accept the model conditions on HuggingFace:
  https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2

> **License note:** DiariZen weights are CC BY-NC 4.0 (non-commercial).
> For commercial use, fall back to `--model community-1` (MIT license).

## HuggingFace token

All three diarization models are gated on HuggingFace.

1. Create a token at https://huggingface.co/settings/tokens
2. Accept each model's terms on its HF page
3. Pass it via:

```bash
export HF_TOKEN=hf_your_token_here
# or use --hf-token hf_your_token_here on each call
```

## vLLM Whisper server

The agent expects Whisper running at `http://localhost:8001` via vLLM:

```bash
vllm serve openai/whisper-large-v3 \
    --port 8001 \
    --dtype bfloat16
```

---

## Usage

```bash
# Basic — local file, DiariZen (default), Persian
python agent.py meeting.mp4

# YouTube link
python agent.py "https://youtu.be/XXXXXXXXX"

# Choose a different diarization model
python agent.py meeting.mp4 --model community-1
python agent.py meeting.mp4 --model pyannote-3.1

# Save outputs
python agent.py meeting.mp4 \
    --out-txt  meeting.txt  \
    --out-json meeting.json \
    --out-srt  meeting.srt

# Non-Persian audio
python agent.py meeting.mp4 --language en

# Full example
python agent.py "https://youtu.be/XXXXXXXXX" \
    --model community-1 \
    --language fa \
    --hf-token hf_... \
    --out-srt  output.srt \
    --out-json output.json
```

## Output formats

| Format | Description |
|--------|-------------|
| Terminal | Colored transcript printed to stdout |
| `--out-txt`  | `[MM:SS]  SPEAKER_XX: text` |
| `--out-json` | Array of `{start, end, duration, speaker, text}` |
| `--out-srt`  | Standard subtitle file for any video player |

## Model comparison (DER — lower is better)

| Model         | AMI meetings | VoxConverse | License   |
|---------------|-------------|-------------|-----------|
| DiariZen v2   | **13.9%**   | **9.1%**   | CC BY-NC 4.0 |
| community-1   | 17.0%       | 11.2%       | MIT       |
| pyannote 3.1  | 18.8%       | 11.2%       | MIT       |
