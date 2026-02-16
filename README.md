# Video Synopsis Generator

Generate condensed video summaries by detecting, tracking, and intelligently rearranging moving objects onto a single timeline — turning hours of surveillance footage into minutes of activity.

## How It Works

```
Input Video (10 min)                    Synopsis (2 min)
┌──────────────────────┐               ┌──────────────────────┐
│                      │               │  🚶       🚗         │
│       🚗             │               │    🚶  🚗            │
│                      │    ──────►    │         🚗  🚶       │
│            🚶        │               │  Multiple objects     │
│                      │               │  shown simultaneously │
└──────────────────────┘               └──────────────────────┘
 Objects appear one                     Objects that were far
 at a time over 10 min                  apart play in parallel
```

The system compresses video by overlapping activities that originally happened at different times but in different parts of the frame — while guaranteeing they never visually collide.

---

## Pipeline

The synopsis is generated in **6 stages**:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Load     │───►│  2. Extract  │───►│  3. Detect   │
│  Video       │    │  Background  │    │  & Track     │
│              │    │  (median)    │    │  (YOLOv8 +   │
│  VideoProc.  │    │  BackgroundE.│    │   Kalman)    │
└─────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  6. Render   │◄──│  5. Optimize │◄──│  4. Generate  │
│  Synopsis    │    │  Placement   │    │  Tubes       │
│              │    │  (Spatial    │    │              │
│  Renderer    │    │   Lanes)     │    │  TubeGen.    │
└─────────────┘    └──────────────┘    └──────────────┘
```

### Stage 1 — Load Video
Read all frames from the input video using OpenCV.

### Stage 2 — Extract Background
Compute a clean background image (median of sampled frames). This is used as the canvas for the synopsis.

### Stage 3 — Detect & Track Objects
- **Detection**: YOLOv8 identifies objects (people, cars, etc.) in each frame. Supports both bounding-box and instance segmentation modes.
- **Tracking**: Kalman filter + Hungarian algorithm maintains consistent track IDs across frames.
- **Segmentation** (optional): Pixel-level masks for seamless compositing.

### Stage 4 — Generate Tubes
A **tube** is one object's complete journey through the video — all its bounding boxes, image patches, and masks bundled together. Filters applied:
- **Minimum duration** — removes brief flickers (< 15 frames)
- **Stationary filter** — removes objects that don't move (parked cars, background objects)
- **Trailing stillness trim** — cuts frames where an object has stopped moving
- **Length cap** — limits tubes to 300 frames (10s), keeping the highest-motion window

### Stage 5 — Optimize Placement (Spatial Lane Optimizer)
The core algorithm that decides **when** each tube appears in the synopsis:

1. **Lane Assignment** — Divide the frame into N vertical strips (default 5). Each tube assigned to the lane matching its average horizontal position. _This is a scheduling heuristic only — objects keep their natural positions._
2. **Intra-Lane Scheduling** — Within each lane, pack tubes sequentially (longest first) with **zero temporal overlap**.
3. **Cross-Lane Conflict Check** — Verify that simultaneously-playing tubes from different lanes don't overlap in pixel space (bbox IoU). If they do, shift the shorter one.

**Guarantees**: Deterministic output, zero visual overlaps.

### Stage 6 — Render
Composite each tube's image patches onto the background canvas at their scheduled times. Supports mask-based blending with edge feathering for seamless results.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) (for video I/O)

### Installation

```bash
git clone https://github.com/AryanManojKumar/Video-Synopsis.git
cd Video-Synopsis
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage
python3.11 main.py input_video.mp4

# Custom output path and compression ratio
python3.11 main.py input_video.mp4 -o output.mp4 -c 0.3

# Disable segmentation (faster, bbox-only mode)
python3.11 main.py input_video.mp4 --no-segmentation

# Disable metadata overlay (no bounding boxes / labels on output)
python3.11 main.py input_video.mp4 --no-metadata
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `input` | _(required)_ | Path to input video |
| `-o, --output` | `synopsis.mp4` | Output video path |
| `-c, --compression` | `0.3` | Target compression ratio (0–1). Lower = shorter synopsis |
| `--no-metadata` | `false` | Disable bounding box / label overlay on output |
| `--no-segmentation` | `false` | Use bbox-only mode (faster, no masks) |

---

## Output: Synopsis Layout Summary

After every run, a human-readable summary is printed:

```
========================================================================
  SYNOPSIS LAYOUT SUMMARY
========================================================================

  #    Track    Class        Original         Synopsis           Dur
  ─────────────────────────────────────────────────────────────────
  1    #2      car          176.8s → 186.8s  0.0s → 10.0s       300
  2    #86     person       203.2s → 216.9s  0.0s → 7.9s        237
  3    #74     person       170.5s → 177.6s  7.9s → 14.8s       207

  Timeline (each █ ≈ 14 frames):
  ──────────────────────────────────────────────────────────────
  #2 car       |█████████████████████                           |
  #86 person   |████████████████                                |
  #74 person   |                ████████████████                |
               0                                               28.2s

  Original : 42.0s (1259 frames)
  Synopsis : 28.2s (846 frames)
  Ratio    : 67.2% compression
  Objects  : 7 tubes placed
========================================================================
```

**Column definitions:**

| Column | Meaning |
|---|---|
| **Original** | When the object appeared in the **source video** (from → to) |
| **Synopsis** | Where it was placed in the **output video** (from → to) |
| **Dur** | Duration in frames |
| **Timeline** | Visual bar chart of each tube's placement |

The Original timestamps enable a future feature: clicking a row in a web UI can seek the source video to that exact moment.

---

## Project Structure

```
Video-Synopsis/
├── main.py                  # CLI entry point — runs the full pipeline
├── app.py                   # FastAPI offline mode (direct processing)
├── core/
│   ├── detector.py          # YOLOv8 object detection + segmentation
│   ├── tracker.py           # Kalman filter multi-object tracker
│   ├── background.py        # Median/mean background extraction
│   ├── tubes.py             # Tube dataclass + TubeGenerator
│   ├── optimizer.py         # SpatialLaneOptimizer (layout scheduling)
│   └── renderer.py          # Video rendering with mask compositing
├── config/
│   └── settings.py          # All configurable parameters
├── api/
│   ├── main.py              # FastAPI app with CORS
│   ├── routes/synopsis.py   # API endpoints for async processing
│   └── schemas.py           # Pydantic request/response models
├── workers/
│   ├── celery_app.py        # Celery configuration
│   └── tasks.py             # Async video processing task
├── db/
│   ├── models.py            # SQLAlchemy Job model
│   └── session.py           # Database session management
├── utils/
│   ├── video.py             # Video loading utilities
│   └── storage.py           # S3/MinIO file storage
├── docker-compose.yml       # Full stack: API + Worker + Postgres + Redis + MinIO
├── Dockerfile               # CUDA-enabled container
├── requirements.txt         # All dependencies
└── requirements-minimal.txt # Core dependencies only
```

---

## Configuration

All parameters are in [`config/settings.py`](config/settings.py) and can be overridden via `.env`:

### Detection & Tracking

| Parameter | Default | Description |
|---|---|---|
| `detection_conf` | `0.5` | YOLO confidence threshold |
| `tracking_max_age` | `30` | Frames before a lost track is deleted |
| `tracking_min_hits` | `3` | Min detections before a track is confirmed |
| `iou_threshold` | `0.3` | IoU threshold for detection-to-track matching |

### Synopsis

| Parameter | Default | Description |
|---|---|---|
| `compression_ratio` | `0.3` | Target synopsis length as fraction of original |
| `min_object_duration` | `15` | Minimum frames for a tube to be included |
| `max_synopsis_duration` | `300` | Maximum synopsis length in seconds |
| `num_spatial_lanes` | `5` | Number of spatial lanes for layout scheduling |

### Tube Filtering

| Parameter | Default | Description |
|---|---|---|
| `max_tube_length` | `300` | Max frames per tube (10s at 30fps) |
| `min_motion_threshold` | `0.08` | Min displacement/size ratio to be "moving" |

### Segmentation

| Parameter | Default | Description |
|---|---|---|
| `use_segmentation` | `true` | Enable instance segmentation masks |
| `seg_model` | `yolov8n-seg` | YOLO segmentation model variant |
| `feather_edges` | `true` | Soften mask edges for blending |
| `feather_radius` | `3` | Gaussian blur kernel for edge feathering |

---

## Running Modes

### 1. CLI (Local)

```bash
python3.11 main.py video.mp4 -o synopsis.mp4 -c 0.3
```

### 2. Offline API

```bash
python3.11 app.py
# Then POST a video to http://localhost:8000/synopsis
```

### 3. Full Stack (Docker)

```bash
cp .env.example .env
docker-compose up --build
```

This starts:
- **API server** (FastAPI + Uvicorn) on port 8000
- **Celery worker** for async video processing
- **PostgreSQL** for job tracking
- **Redis** as Celery broker
- **MinIO** for S3-compatible file storage

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/synopsis` | Upload video, start async processing |
| `GET` | `/api/v1/synopsis/{job_id}` | Check job status and progress |
| `GET` | `/api/v1/synopsis/{job_id}/download` | Download completed synopsis |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | Kalman Filter + Hungarian Algorithm |
| Segmentation | YOLOv8-seg instance segmentation |
| Background | Median frame extraction |
| Layout Optimizer | Spatial Lane Optimizer (deterministic) |
| Video I/O | OpenCV |
| API Framework | FastAPI |
| Task Queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy |
| Object Storage | MinIO (S3-compatible) |
| Container | Docker + NVIDIA CUDA |
