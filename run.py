#!/usr/bin/env python3
import sys
from pathlib import Path
from main import process_video_synopsis

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <video_path> [output_path] [compression_ratio]")
        print("\nExample:")
        print("  python run.py input.mp4")
        print("  python run.py input.mp4 output.mp4")
        print("  python run.py input.mp4 output.mp4 0.5")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "synopsis.mp4"
    compression = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    process_video_synopsis(video_path, output_path, compression)

if __name__ == "__main__":
    main()
