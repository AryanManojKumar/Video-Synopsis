import cv2
import numpy as np
from typing import List

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
    
    def get_fps(self) -> int:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return int(fps) if fps > 0 else 30
    
    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_resolution(self) -> tuple:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def extract_frames(self, max_frames: int = None) -> List[np.ndarray]:
        frames = []
        count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frames.append(frame)
            count += 1
            
            if max_frames and count >= max_frames:
                break
        
        self.cap.release()
        return frames
    
    def extract_frames_generator(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
        self.cap.release()
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
