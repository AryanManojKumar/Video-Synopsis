from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uuid
from main import process_video_synopsis

app = FastAPI(title="Video Synopsis - Offline Mode")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Video Synopsis API - Offline Mode", "status": "ready"}

@app.post("/synopsis")
async def create_synopsis(
    file: UploadFile = File(...),
    compression_ratio: float = 0.3
):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Invalid video format. Use .mp4, .avi, or .mov")
    
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    output_path = OUTPUT_DIR / f"{job_id}_synopsis.mp4"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        process_video_synopsis(
            str(input_path),
            str(output_path),
            compression_ratio,
            add_metadata=True
        )
        
        return {
            "job_id": job_id,
            "status": "completed",
            "download_url": f"/download/{job_id}"
        }
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        input_path.unlink(missing_ok=True)

@app.get("/download/{job_id}")
async def download_synopsis(job_id: str):
    output_path = OUTPUT_DIR / f"{job_id}_synopsis.mp4"
    
    if not output_path.exists():
        raise HTTPException(404, "Synopsis not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"synopsis_{job_id}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
