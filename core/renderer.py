import cv2
import numpy as np
from typing import List, Tuple
from core.tubes import Tube

class SynopsisRenderer:
    def __init__(self, background: np.ndarray, fps: int = 30):
        self.background = background
        self.fps = fps
        self.height, self.width = background.shape[:2]
        
    def render(self, placements: List[Tuple[Tube, int]], 
              output_path: str, add_metadata: bool = True):
        if not placements:
            raise ValueError("No placements to render")
        
        max_frame = max([start_time + len(tube.bboxes) 
                        for tube, start_time in placements])
        
        # Trim trailing dead frames: find last frame with active tube activity
        last_active_frame = 0
        for tube, start_time in placements:
            tube_end = start_time + len(tube.bboxes)
            if tube_end > last_active_frame:
                last_active_frame = tube_end
        
        # Add 1-second buffer after last activity, but don't exceed original max
        buffer_frames = self.fps
        max_frame = min(max_frame, last_active_frame + buffer_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        for frame_idx in range(max_frame):
            canvas = self.background.copy()
            
            for tube, start_time in placements:
                local_idx = frame_idx - start_time
                
                if 0 <= local_idx < len(tube.bboxes):
                    bbox = tube.bboxes[local_idx]
                    frame = tube.frames[local_idx]
                    
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.width, x2), min(self.height, y2)
                    
                    if x2 > x1 and y2 > y1:
                        patch = frame[y1:y2, x1:x2]
                        if patch.shape[0] > 0 and patch.shape[1] > 0:
                            canvas[y1:y2, x1:x2] = patch
                        
                        if add_metadata:
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{tube.class_name} #{tube.track_id}"
                            cv2.putText(canvas, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if add_metadata:
                timestamp = f"Frame: {frame_idx}/{max_frame}"
                cv2.putText(canvas, timestamp, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(canvas)
        
        out.release()
        return output_path
    
    def render_with_blending(self, placements: List[Tuple[Tube, int]], 
                            output_path: str, alpha: float = 0.7):
        if not placements:
            raise ValueError("No placements to render")
        
        max_frame = max([start_time + len(tube.bboxes) 
                        for tube, start_time in placements])
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        for frame_idx in range(max_frame):
            canvas = self.background.copy().astype(float)
            
            for tube, start_time in placements:
                local_idx = frame_idx - start_time
                
                if 0 <= local_idx < len(tube.bboxes):
                    bbox = tube.bboxes[local_idx]
                    frame = tube.frames[local_idx]
                    
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.width, x2), min(self.height, y2)
                    
                    if x2 > x1 and y2 > y1:
                        patch = frame[y1:y2, x1:x2].astype(float)
                        if patch.shape[0] > 0 and patch.shape[1] > 0:
                            canvas[y1:y2, x1:x2] = (alpha * patch + 
                                                   (1 - alpha) * canvas[y1:y2, x1:x2])
            
            out.write(canvas.astype(np.uint8))
        
        out.release()
        return output_path
