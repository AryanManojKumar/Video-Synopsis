from pydantic import BaseModel
from typing import Optional

class SynopsisRequest(BaseModel):
    compression_ratio: float = 0.3
    use_genetic: bool = False
    min_object_duration: int = 30

class SynopsisResponse(BaseModel):
    job_id: str
    task_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result_url: Optional[str] = None
    error: Optional[str] = None
