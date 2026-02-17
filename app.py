import json
import shutil
import subprocess
import uuid
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from main import process_video_synopsis

app = FastAPI(title="Video Synopsis")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracker
jobs: dict = {}


@app.get("/api/health")
async def health():
    return {"status": "ready"}


@app.post("/api/synopsis")
async def create_synopsis(
    file: UploadFile = File(...),
    compression_ratio: float = Form(0.3),
):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Invalid video format. Use .mp4, .avi, or .mov")

    job_id = str(uuid.uuid4())[:8]
    input_path = UPLOAD_DIR / f"{job_id}_original.mp4"
    browser_original = UPLOAD_DIR / f"{job_id}_original_h264.mp4"
    output_path = OUTPUT_DIR / f"{job_id}_synopsis.mp4"
    json_path = OUTPUT_DIR / f"{job_id}_synopsis.mp4.json"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Re-encode original to H.264 for browser playback
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(browser_original)],
            check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: just copy the original as-is
        shutil.copy2(input_path, browser_original)

    jobs[job_id] = {
        "status": "processing",
        "progress": "Starting...",
        "input_path": str(input_path),
        "browser_original": str(browser_original),
        "output_path": str(output_path),
        "json_path": str(json_path),
        "error": None,
    }

    # Run processing in a background thread
    def run():
        try:
            process_video_synopsis(
                str(input_path),
                str(output_path),
                compression_ratio,
                add_metadata=True,
                use_segmentation=True,
            )
            # Load the generated JSON summary
            if json_path.exists():
                with open(json_path) as f:
                    jobs[job_id]["summary"] = json.load(f)
            jobs[job_id]["status"] = "completed"
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    resp = {"status": job["status"]}
    if job["status"] == "completed":
        resp["summary"] = job.get("summary", {})
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


@app.get("/api/video/original/{job_id}")
async def stream_original(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    path = Path(jobs[job_id].get("browser_original", jobs[job_id]["input_path"]))
    if not path.exists():
        raise HTTPException(404, "Original video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/video/synopsis/{job_id}")
async def stream_synopsis(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    path = Path(jobs[job_id]["output_path"])
    if not path.exists():
        raise HTTPException(404, "Synopsis not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/summary/{job_id}")
async def get_summary(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    json_path = Path(jobs[job_id]["json_path"])
    if not json_path.exists():
        raise HTTPException(404, "Summary not found")
    with open(json_path) as f:
        return json.load(f)


# Serve frontend static files — must be LAST (catch-all mount)
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
