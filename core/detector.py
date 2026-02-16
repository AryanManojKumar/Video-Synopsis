import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
from config import settings


class ObjectDetector:
    def __init__(self, model_path: str = None, use_segmentation: bool = None):
        self.device = 'cuda' if settings.gpu_enabled and torch.cuda.is_available() else 'cpu'
        self.use_segmentation = use_segmentation if use_segmentation is not None else settings.use_segmentation
        
        if model_path:
            pass  # Use explicitly provided path
        elif self.use_segmentation:
            model_path = f"{settings.model_path}/{settings.seg_model}.pt"
        else:
            model_path = f"{settings.model_path}/yolov8n.pt"
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = settings.detection_conf
    
    def _create_mask_from_polygon(self, polygon_xy, frame_h: int, frame_w: int) -> np.ndarray:
        """Create a sharp binary mask by drawing the segmentation polygon 
        at full frame resolution. This avoids the blurry result from
        resizing the low-res proto mask tensor."""
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if polygon_xy is not None and len(polygon_xy) > 0:
            pts = polygon_xy.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
        return mask
        
    def detect(self, frame: np.ndarray) -> List[dict]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        
        frame_h, frame_w = frame.shape[:2]
        has_masks = self.use_segmentation and results.masks is not None
        
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': cls,
                'class_name': results.names[cls],
                'mask': None
            }
            
            # Create sharp mask from polygon contours at full resolution
            if has_masks and i < len(results.masks.xy):
                detection['mask'] = self._create_mask_from_polygon(
                    results.masks.xy[i], frame_h, frame_w
                )
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        results = self.model(frames, conf=self.conf_threshold, verbose=False)
        batch_detections = []
        
        for frame_idx, result in enumerate(results):
            frame_detections = []
            frame_h, frame_w = frames[frame_idx].shape[:2]
            has_masks = self.use_segmentation and result.masks is not None
            
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': result.names[cls],
                    'mask': None
                }
                
                if has_masks and i < len(result.masks.xy):
                    detection['mask'] = self._create_mask_from_polygon(
                        result.masks.xy[i], frame_h, frame_w
                    )
                
                frame_detections.append(detection)
            batch_detections.append(frame_detections)
        
        return batch_detections
