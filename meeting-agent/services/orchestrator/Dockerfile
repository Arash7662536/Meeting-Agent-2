# pytorch/pytorch already has: Python 3.11, PyTorch 2.4.0, CUDA 12.1, cudnn8, conda
# We use conda-forge for ffmpeg/libsndfile to avoid Ubuntu apt mirrors (unreachable on some hosts)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN conda install -y -c conda-forge ffmpeg libsndfile curl ca-certificates \
    && conda clean --all --yes

# torchaudio matching the installed torch — --no-deps avoids reinstalling torch
# demucs needs torchaudio at runtime
RUN pip install --no-cache-dir --no-deps \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
