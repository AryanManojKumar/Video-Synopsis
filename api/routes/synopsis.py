from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from workers.tasks import process_video_synopsis
from api.schemas import SynopsisRequest, SynopsisResponse, JobStatus
from utils.storage import upload_to_s3, download_from_s3
import uuid

router = APIRouter()

@router.post("/synopsis", response_model=SynopsisResponse)
async def create_synopsis(
    file: UploadFile = File(...),
    compression_ratio: float = 0.3,
    use_genetic: bool = False
):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Invalid video format")
    
    job_id = str(uuid.uuid4())
    video_path = f"uploads/{job_id}/{file.filename}"
    
    contents = await file.read()
    s3_url = upload_to_s3(contents, video_path)
    
    task = process_video_synopsis.delay(
        job_id=job_id,
        video_path=video_path,
        compression_ratio=compression_ratio,
        use_genetic=use_genetic
    )
    
    return SynopsisResponse(
        job_id=job_id,
        task_id=task.id,
        status="processing",
        message="Video synopsis generation started"
    )

@router.get("/synopsis/{job_id}", response_model=JobStatus)
async def get_synopsis_status(job_id: str):
    from workers.celery_app import celery_app
    from db.models import Job
    from db.session import get_db
    
    db = next(get_db())
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        result_url=job.result_url,
        error=job.error
    )

@router.get("/synopsis/{job_id}/download")
async def download_synopsis(job_id: str):
    from db.models import Job
    from db.session import get_db
    
    db = next(get_db())
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job or job.status != "completed":
        raise HTTPException(404, "Synopsis not ready")
    
    local_path = download_from_s3(job.result_url)
    return FileResponse(local_path, media_type="video/mp4", filename=f"synopsis_{job_id}.mp4")
