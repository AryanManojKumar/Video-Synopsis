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
    tracking_max_age: int = 45  # Increased from 30 for better occlusion handling
    tracking_min_hits: int = 3
    iou_threshold: float = 0.3
    
    # Synopsis configs
    compression_ratio: float = 0.3
    min_object_duration: int = 15
    max_synopsis_duration: int = 300
    num_spatial_lanes: int = 5  # Number of spatial lanes for layout scheduling
    
    # Tube filtering configs
    max_tube_length: int = 300  # Max frames per tube (10s at 30fps)
    min_motion_threshold: float = 0.08  # Min displacement/object-size ratio to be considered "moving"
    max_gap_fill: int = 15  # Max missing frames to interpolate (0.5s at 30fps)
    group_merge_distance: float = 2.0  # Spatial proximity threshold (multiples of avg object diagonal)
    
    # Segmentation configs
    use_segmentation: bool = True
    seg_model: str = "yolov8n-seg"  # Model variant for segmentation
    feather_edges: bool = True  # Soften mask edges via Gaussian blur
    feather_radius: int = 3  # Gaussian blur kernel size for edge feathering
    
    class Config:
        env_file = ".env"
        extra = "ignore"

try:
    settings = Settings()
except Exception:
    settings = Settings(_env_file=None)
