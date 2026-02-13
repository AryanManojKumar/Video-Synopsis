from fastapi import APIRouter
import torch
from config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_enabled": settings.gpu_enabled
    }

@router.get("/gpu")
async def gpu_status():
    if torch.cuda.is_available():
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0)
        }
    return {"available": False}
