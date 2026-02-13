import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
from config import settings

class ObjectDetector:
    def __init__(self, model_path: str = None):
        self.device = 'cuda' if settings.gpu_enabled and torch.cuda.is_available() else 'cpu'
        model_path = model_path or f"{settings.model_path}/yolov8n.pt"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = settings.detection_conf
        
    def detect(self, frame: np.ndarray) -> List[dict]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': cls,
                'class_name': results.names[cls]
            })
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        results = self.model(frames, conf=self.conf_threshold, verbose=False)
        batch_detections = []
        
        for result in results:
            frame_detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                frame_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': result.names[cls]
                })
            batch_detections.append(frame_detections)
        
        return batch_detections
