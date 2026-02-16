import cv2
import numpy as np
from typing import List, Tuple
from core.tubes import Tube
from config import settings


class SynopsisRenderer:
    def __init__(self, background: np.ndarray, fps: int = 30):
        self.background = background
        self.fps = fps
        self.height, self.width = background.shape[:2]
        self.feather_edges = settings.feather_edges
        self.feather_radius = settings.feather_radius
        
    def _composite_object(self, canvas: np.ndarray, frame: np.ndarray,
                          bbox: List[int], mask: np.ndarray = None):
        """Composite an object onto the canvas using mask or bbox fallback."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        patch = frame[y1:y2, x1:x2]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            return
        
        # Validate mask dimensions match the frame
        use_mask = (mask is not None and 
                    mask.shape[0] == frame.shape[0] and 
                    mask.shape[1] == frame.shape[1])
        
        if use_mask:
            # Mask-based compositing: only paste object pixels
            mask_crop = mask[y1:y2, x1:x2].astype(np.float32)
            
            # Edge-only feathering: soften just the boundary, not the whole mask
            if self.feather_edges and self.feather_radius > 0:
                k = self.feather_radius | 1  # Ensure odd kernel size
                # Erode to find interior, blur only the edge band
                eroded = cv2.erode(mask_crop, np.ones((3, 3), np.uint8), iterations=1)
                edge_band = mask_crop - eroded  # 1 only at edges
                blurred_edges = cv2.GaussianBlur(mask_crop, (k, k), 0)
                # Keep solid interior, blend only at edges
                mask_crop = np.where(edge_band > 0, blurred_edges, mask_crop)
            
            # Expand mask to 3 channels and blend
            mask_3ch = np.stack([mask_crop] * 3, axis=-1)
            canvas_region = canvas[y1:y2, x1:x2].astype(np.float32)
            blended = mask_3ch * patch.astype(np.float32) + (1 - mask_3ch) * canvas_region
            canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            # Fallback: rectangular bbox paste
            canvas[y1:y2, x1:x2] = patch
        
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
                    mask = tube.masks[local_idx] if tube.masks else None
                    
                    self._composite_object(canvas, frame, bbox, mask)
                        
                    if add_metadata:
                        x1, y1, x2, y2 = bbox
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(self.width, x2), min(self.height, y2)
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
        
        # Trim trailing dead frames
        last_active_frame = 0
        for tube, start_time in placements:
            tube_end = start_time + len(tube.bboxes)
            if tube_end > last_active_frame:
                last_active_frame = tube_end
        buffer_frames = self.fps
        max_frame = min(max_frame, last_active_frame + buffer_frames)
        
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
                    mask = tube.masks[local_idx] if tube.masks else None
                    
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.width, x2), min(self.height, y2)
                    
                    if x2 > x1 and y2 > y1:
                        patch = frame[y1:y2, x1:x2].astype(float)
                        use_mask = (mask is not None and
                                    mask.shape[0] == frame.shape[0] and
                                    mask.shape[1] == frame.shape[1])
                        if patch.shape[0] > 0 and patch.shape[1] > 0:
                            if use_mask:
                                mask_crop = mask[y1:y2, x1:x2].astype(float)
                                if self.feather_edges and self.feather_radius > 0:
                                    k = self.feather_radius | 1
                                    eroded = cv2.erode(mask_crop.astype(np.float32), np.ones((3,3), np.uint8), iterations=1)
                                    edge_band = mask_crop - eroded
                                    blurred_edges = cv2.GaussianBlur(mask_crop.astype(np.float32), (k, k), 0)
                                    mask_crop = np.where(edge_band > 0, blurred_edges, mask_crop)
                                mask_3ch = np.stack([mask_crop] * 3, axis=-1)
                                canvas[y1:y2, x1:x2] = (
                                    alpha * mask_3ch * patch +
                                    (1 - alpha * mask_3ch) * canvas[y1:y2, x1:x2]
                                )
                            else:
                                canvas[y1:y2, x1:x2] = (
                                    alpha * patch + 
                                    (1 - alpha) * canvas[y1:y2, x1:x2]
                                )
            
            out.write(canvas.astype(np.uint8))
        
        out.release()
        return output_path
