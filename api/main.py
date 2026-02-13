from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from api.routes import synopsis, health
from config import settings

app = FastAPI(title="Video Synopsis API", version="1.0.0")

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(synopsis.router, prefix="/api/v1", tags=["synopsis"])

@app.get("/")
async def root():
    return {"message": "Video Synopsis API", "version": "1.0.0"}
