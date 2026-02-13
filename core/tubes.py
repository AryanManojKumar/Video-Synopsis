import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from config import settings

@dataclass
class Tube:
    track_id: int
    class_name: str
    start_frame: int
    end_frame: int
    bboxes: List[List[int]]
    frames: List[np.ndarray]
    _spatial_bounds: tuple = None
    
    @property
    def duration(self):
        return len(self.bboxes)
    
    @property
    def center_trajectory(self):
        centers = []
        for bbox in self.bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))
        return centers
    
    @property
    def spatial_bounds(self):
        """Get overall bounding box covering entire trajectory"""
        if self._spatial_bounds is None:
            if not self.bboxes:
                self._spatial_bounds = (0, 0, 0, 0)
            else:
                x_min = min(bbox[0] for bbox in self.bboxes)
                y_min = min(bbox[1] for bbox in self.bboxes)
                x_max = max(bbox[2] for bbox in self.bboxes)
                y_max = max(bbox[3] for bbox in self.bboxes)
                self._spatial_bounds = (x_min, y_min, x_max, y_max)
        return self._spatial_bounds

class TubeGenerator:
    def __init__(self):
        self.min_duration = settings.min_object_duration
        
    def generate_tubes(self, tracks_per_frame: List[List[dict]], 
                       video_frames: List[np.ndarray]) -> List[Tube]:
        track_data = {}
        
        for frame_idx, tracks in enumerate(tracks_per_frame):
            for track in tracks:
                tid = track['track_id']
                if tid not in track_data:
                    track_data[tid] = {
                        'class_name': track['class_name'],
                        'frames': [],
                        'bboxes': [],
                        'frame_indices': []
                    }
                
                track_data[tid]['frames'].append(video_frames[frame_idx])
                track_data[tid]['bboxes'].append(track['bbox'])
                track_data[tid]['frame_indices'].append(frame_idx)
        
        tubes = []
        for tid, data in track_data.items():
            if len(data['frames']) < self.min_duration:
                continue
            
            tube = Tube(
                track_id=tid,
                class_name=data['class_name'],
                start_frame=data['frame_indices'][0],
                end_frame=data['frame_indices'][-1],
                bboxes=data['bboxes'],
                frames=data['frames']
            )
            tubes.append(tube)
        
        return tubes
    
    def extract_object_patches(self, tube: Tube) -> List[np.ndarray]:
        patches = []
        for frame, bbox in zip(tube.frames, tube.bboxes):
            x1, y1, x2, y2 = bbox
            patch = frame[y1:y2, x1:x2]
            patches.append(patch)
        return patches
