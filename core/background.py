import cv2
import numpy as np
from typing import List

class BackgroundExtractor:
    def __init__(self, method='median'):
        self.method = method
        
    def extract(self, video_path: str, sample_rate: int = 30) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
            
            if len(frames) >= 50:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        if self.method == 'median':
            background = np.median(frames, axis=0).astype(np.uint8)
        elif self.method == 'mean':
            background = np.mean(frames, axis=0).astype(np.uint8)
        else:
            background = frames[0]
        
        return background
    
    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        if self.method == 'median':
            background = np.median(frames, axis=0).astype(np.uint8)
        elif self.method == 'mean':
            background = np.mean(frames, axis=0).astype(np.uint8)
        else:
            background = frames[0]
        
        return background
