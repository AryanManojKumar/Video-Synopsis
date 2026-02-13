import cv2
import argparse
from pathlib import Path
from core.detector import ObjectDetector
from core.tracker import MultiObjectTracker
from core.background import BackgroundExtractor
from core.tubes import TubeGenerator
from core.optimizer import ConflictResolver
from core.renderer import SynopsisRenderer
from utils.video import VideoProcessor

def process_video_synopsis(video_path: str, output_path: str, 
                          compression_ratio: float = 0.3,
                          use_genetic: bool = False,
                          add_metadata: bool = True):
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
    
    print("Step 3/6: Detecting and tracking objects...")
    detector = ObjectDetector()
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
    print(f"Generated {len(tubes)} tubes")
    
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
    resolver = ConflictResolver(grid_size=(3, 3), overlap_threshold=0.3)
    resolver.compression_ratio = compression_ratio
    
    if use_genetic:
        print("  Using genetic algorithm with spatial zones...")
        placements = resolver.optimize_genetic(tubes, len(frames), width, height)
    else:
        print("  Using greedy algorithm with spatial zones...")
        placements = resolver.optimize_placement(tubes, len(frames), width, height)
    
    print(f"Placed {len(placements)} objects")
    
    print("Step 6/6: Rendering synopsis...")
    renderer = SynopsisRenderer(background, fps)
    renderer.render(placements, output_path, add_metadata=add_metadata)
    
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
    parser.add_argument("-g", "--genetic", action="store_true", help="Use genetic algorithm")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata overlay")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Video file not found: {args.input}")
        exit(1)
    
    process_video_synopsis(
        args.input,
        args.output,
        args.compression,
        args.genetic,
        not args.no_metadata
    )
