import cv2
import json
import subprocess
import os
import argparse
from pathlib import Path
from core.detector import ObjectDetector
from core.tracker import MultiObjectTracker
from core.background import BackgroundExtractor
from core.tubes import TubeGenerator
from core.optimizer import SpatialLaneOptimizer
from core.renderer import SynopsisRenderer
from utils.video import VideoProcessor

def process_video_synopsis(video_path: str, output_path: str, 
                          compression_ratio: float = 0.3,
                          add_metadata: bool = True,
                          use_segmentation: bool = True):
    print(f"Processing video: {video_path}")
    
    print("Step 1/6: Loading video...")
    video_proc = VideoProcessor(video_path)
    frames = video_proc.extract_frames()
    fps = video_proc.get_fps()
    width, height = video_proc.get_resolution()
    
    if fps == 0:
        fps = 30
        print(f"Warning: Could not detect FPS, using default {fps} FPS")
    
    if width == 0 or height == 0:
        if len(frames) > 0:
            height, width = frames[0].shape[:2]
            print(f"Warning: Could not detect resolution from video metadata, using frame dimensions")
        else:
            width, height = 1920, 1080
            print(f"Warning: Using default resolution {width}x{height}")
    
    print(f"Loaded {len(frames)} frames at {fps} FPS ({width}x{height})")
    
    print("Step 2/6: Extracting background...")
    bg_extractor = BackgroundExtractor()
    background = bg_extractor.extract_from_frames(frames[::30])
    
    seg_mode = "segmentation" if use_segmentation else "bbox-only"
    print(f"Step 3/6: Detecting and tracking objects ({seg_mode} mode)...")
    detector = ObjectDetector(use_segmentation=use_segmentation)
    tracker = MultiObjectTracker()
    
    tracks_per_frame = []
    for i, frame in enumerate(frames):
        detections = detector.detect(frame)
        tracks = tracker.update(detections, i)
        tracks_per_frame.append(tracks)
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(frames)} frames")
    
    print(f"Step 4/6: Generating tubes...")
    tube_gen = TubeGenerator()
    tubes = tube_gen.generate_tubes(tracks_per_frame, frames)
    stats = tube_gen.filter_stats
    print(f"Generated {stats['total']} raw tubes → {len(tubes)} after filtering")
    if stats['stationary_removed'] > 0:
        print(f"  Removed {stats['stationary_removed']} stationary tubes:")
        for detail in stats.get('removed_details', []):
            print(f"    - Track #{detail['track_id']} ({detail['class']}, "
                  f"{detail['duration']} frames, motion={detail['motion_score']}, "
                  f"threshold={detail['threshold']})")
    if stats['trimmed'] > 0:
        print(f"  Trimmed trailing stillness from {stats['trimmed']} tubes")
    if stats['capped'] > 0:
        print(f"  Capped {stats['capped']} tubes to max length")
    
    if len(tubes) == 0:
        print("Warning: No objects tracked long enough. Try lowering min_object_duration.")
        print("Creating empty synopsis with background only...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (background.shape[1], background.shape[0]))
        out.write(background)
        out.release()
        
        print(f"Output saved to: {output_path}")
        return
    
    print("Step 5/6: Optimizing placement...")
    optimizer = SpatialLaneOptimizer(overlap_threshold=0.3)
    optimizer.compression_ratio = compression_ratio
    
    print("  Using spatial-lane deterministic optimizer...")
    placements = optimizer.optimize(tubes, len(frames), width, height)
    
    print("Step 6/6: Rendering synopsis...")
    renderer = SynopsisRenderer(background, fps)
    renderer.render(placements, output_path, add_metadata=add_metadata)
    
    # Re-encode to H.264 for browser compatibility
    tmp_path = output_path + ".tmp.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", tmp_path],
            check=True, capture_output=True
        )
        os.replace(tmp_path, output_path)
        print("  Re-encoded to H.264 (browser-compatible)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print("  Warning: ffmpeg not available, video may not play in browsers")
    
    # Print human-readable layout summary
    optimizer.print_summary(placements, fps, len(frames))
    
    # Save structured summary as JSON sidecar
    summary_data = optimizer.get_summary_data(placements, fps, len(frames))
    json_path = Path(output_path).with_suffix('.mp4.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to: {json_path}")
    
    original_duration = len(frames) / fps
    synopsis_duration = max([start + tube.duration for tube, start in placements]) / fps if placements else 0
    
    print(f"\nCompleted!")
    print(f"Original duration: {original_duration:.2f}s")
    print(f"Synopsis duration: {synopsis_duration:.2f}s")
    if original_duration > 0:
        print(f"Compression ratio: {synopsis_duration/original_duration:.2%}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Synopsis Generator")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", default="synopsis.mp4", help="Output video path")
    parser.add_argument("-c", "--compression", type=float, default=0.3, help="Compression ratio (0-1)")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata overlay")
    parser.add_argument("--no-segmentation", action="store_true", help="Disable segmentation (use bbox-only mode)")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Video file not found: {args.input}")
        exit(1)
    
    process_video_synopsis(
        args.input,
        args.output,
        args.compression,
        not args.no_metadata,
        not args.no_segmentation
    )
