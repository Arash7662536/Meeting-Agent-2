from redis import Redis
from rq import Queue, Worker

from .db import init_db
from .settings import settings


def main() -> None:
    init_db()
    conn = Redis.from_url(settings.REDIS_URL)
    Worker([Queue("meeting-agent", connection=conn)], connection=conn).work()


if __name__ == "__main__":
    main()
