import uuid
from typing import Optional

from redis import Redis
from rq import Queue

from .db import Job, SessionLocal, init_db
from .pipeline import run_pipeline
from .settings import settings

redis_conn = Redis.from_url(settings.REDIS_URL)
queue = Queue("meeting-agent", connection=redis_conn, default_timeout=7200)


def enqueue_job(
    input_path: str,
    is_url: bool,
    input_name: str,
    denoise: bool = False,
    identify_speakers: bool = True,
    num_speakers: Optional[int] = None,
    language: Optional[str] = None,
) -> str:
    init_db()
    job_id = uuid.uuid4().hex[:12]
    source = "url" if is_url else "upload"

    with SessionLocal() as db:
        db.add(Job(
            id=job_id, source=source, input_name=input_name, status="queued"
        ))
        db.commit()

    queue.enqueue(
        run_pipeline,
        job_id, input_path, is_url,
        denoise, identify_speakers, num_speakers, language,
        job_id=job_id,
    )
    return job_id
