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
        self.max_tube_length = settings.max_tube_length
        self.min_motion_threshold = settings.min_motion_threshold
        self._stats = {'total': 0, 'stationary_removed': 0, 'capped': 0, 'trimmed': 0}
        
    @property
    def filter_stats(self):
        return self._stats
        
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
        
        self._stats['total'] = len(tubes)
        
        # Post-processing: filter and trim tubes
        tubes = self._filter_stationary(tubes)
        tubes = [self._trim_trailing_stillness(t) for t in tubes]
        # Re-filter after trimming (some may be too short now)
        tubes = [t for t in tubes if t.duration >= self.min_duration]
        tubes = self._cap_tube_lengths(tubes)
        
        return tubes
    
    def _compute_displacement(self, tube: Tube) -> float:
        """Compute total center displacement across trajectory."""
        centers = tube.center_trajectory
        if len(centers) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            total += (dx**2 + dy**2) ** 0.5
        return total
    
    def _get_object_diagonal(self, tube: Tube) -> float:
        """Average diagonal size of the object across the tube."""
        diags = []
        for bbox in tube.bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            diags.append((w**2 + h**2) ** 0.5)
        return sum(diags) / len(diags) if diags else 1.0
    
    def _filter_stationary(self, tubes: List[Tube]) -> List[Tube]:
        """Remove tubes where the object barely moves relative to its size."""
        filtered = []
        for tube in tubes:
            displacement = self._compute_displacement(tube)
            diagonal = self._get_object_diagonal(tube)
            
            # Normalize displacement by object size and duration
            # A truly moving object should displace at least threshold * diagonal * sqrt(duration)
            normalized_motion = displacement / (diagonal * max(1.0, len(tube.bboxes) ** 0.5))
            
            # Be gentler on short tubes — brief appearances don't accumulate
            # much displacement even when genuinely moving
            threshold = self.min_motion_threshold
            if len(tube.bboxes) < 60:
                threshold *= 0.5
            
            if normalized_motion >= threshold:
                filtered.append(tube)
            else:
                self._stats['stationary_removed'] += 1
                # Log why this tube was removed for diagnostics
                self._stats.setdefault('removed_details', []).append({
                    'track_id': tube.track_id,
                    'class': tube.class_name,
                    'duration': len(tube.bboxes),
                    'motion_score': round(normalized_motion, 4),
                    'threshold': round(threshold, 4),
                })
        
        return filtered
    
    def _trim_trailing_stillness(self, tube: Tube) -> Tube:
        """Trim frames from the end of a tube where the object has stopped moving."""
        centers = tube.center_trajectory
        if len(centers) < 2:
            return tube
        
        diagonal = self._get_object_diagonal(tube)
        still_threshold = diagonal * 0.02  # 2% of object diagonal per frame
        
        # Scan backwards to find where movement stopped
        last_moving_idx = len(centers) - 1
        for i in range(len(centers) - 1, 0, -1):
            dx = abs(centers[i][0] - centers[i-1][0])
            dy = abs(centers[i][1] - centers[i-1][1])
            frame_displacement = (dx**2 + dy**2) ** 0.5
            
            if frame_displacement > still_threshold:
                last_moving_idx = i
                break
        else:
            # Entire trajectory is still — keep as-is (stationary filter handles this)
            return tube
        
        # Add a small buffer (10 frames) after last movement
        trim_end = min(last_moving_idx + 10, len(tube.bboxes))
        
        if trim_end < len(tube.bboxes):
            self._stats['trimmed'] += 1
            return Tube(
                track_id=tube.track_id,
                class_name=tube.class_name,
                start_frame=tube.start_frame,
                end_frame=tube.start_frame + trim_end - 1,
                bboxes=tube.bboxes[:trim_end],
                frames=tube.frames[:trim_end]
            )
        
        return tube
    
    def _cap_tube_lengths(self, tubes: List[Tube]) -> List[Tube]:
        """Cap tubes to max_tube_length, keeping the highest-motion window."""
        capped = []
        for tube in tubes:
            if len(tube.bboxes) <= self.max_tube_length:
                capped.append(tube)
                continue
            
            self._stats['capped'] += 1
            
            # Find the window of max_tube_length frames with the most motion
            centers = tube.center_trajectory
            best_start = 0
            best_motion = 0.0
            
            for start in range(0, len(centers) - self.max_tube_length + 1):
                window_motion = 0.0
                for i in range(start + 1, start + self.max_tube_length):
                    dx = centers[i][0] - centers[i-1][0]
                    dy = centers[i][1] - centers[i-1][1]
                    window_motion += (dx**2 + dy**2) ** 0.5
                
                if window_motion > best_motion:
                    best_motion = window_motion
                    best_start = start
            
            end = best_start + self.max_tube_length
            capped.append(Tube(
                track_id=tube.track_id,
                class_name=tube.class_name,
                start_frame=tube.start_frame + best_start,
                end_frame=tube.start_frame + end - 1,
                bboxes=tube.bboxes[best_start:end],
                frames=tube.frames[best_start:end]
            ))
        
        return capped
    
    def extract_object_patches(self, tube: Tube) -> List[np.ndarray]:
        patches = []
        for frame, bbox in zip(tube.frames, tube.bboxes):
            x1, y1, x2, y2 = bbox
            patch = frame[y1:y2, x1:x2]
            patches.append(patch)
        return patches
