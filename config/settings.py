from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Optional cloud storage
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Core settings
    model_path: str = "./models/weights"
    gpu_enabled: bool = True
    max_workers: int = 4
    api_port: int = 8000
    
    # Model configs
    detection_conf: float = 0.5
    tracking_max_age: int = 30
    tracking_min_hits: int = 3
    iou_threshold: float = 0.3
    
    # Synopsis configs
    compression_ratio: float = 0.3
    min_object_duration: int = 30
    max_synopsis_duration: int = 300
    
    class Config:
        env_file = ".env"
        extra = "ignore"

try:
    settings = Settings()
except Exception:
    settings = Settings(_env_file=None)
