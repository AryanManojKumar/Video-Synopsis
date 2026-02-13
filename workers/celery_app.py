from celery import Celery
from config import settings

celery_app = Celery(
    "video_synopsis",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10
)
