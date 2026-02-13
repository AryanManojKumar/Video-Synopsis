import numpy as np
from typing import List, Dict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from config import settings

class Track:
    count = 0
    
    def __init__(self, bbox, class_id, class_name):
        self.id = Track.count
        Track.count += 1
        self.kf = self._init_kalman(bbox)
        self.class_id = class_id
        self.class_name = class_name
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []
        
    def _init_kalman(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],
                         [0,1,0,0,0,1,0],
                         [0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],
                         [0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0],
                         [0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0]])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self._bbox_to_z(bbox)
        return kf
    
    def _bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([x, y, w, h]).reshape((4, 1))
    
    def _z_to_bbox(self, z):
        w = z[2]
        h = z[3]
        return [z[0]-w/2., z[1]-h/2., z[0]+w/2., z[1]+h/2.]
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x[:4])
    
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))
        self.history.append(bbox)

class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.max_age = settings.tracking_max_age
        self.min_hits = settings.tracking_min_hits
        self.iou_threshold = settings.iou_threshold
        
    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(self, detections: List[dict], frame_idx: int) -> List[dict]:
        for track in self.tracks:
            track.predict()
        
        if len(detections) == 0:
            return self._get_active_tracks()
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            pred_bbox = track._z_to_bbox(track.kf.x[:4])
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._iou(pred_bbox, det['bbox'])
        
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        
        for t, d in matched_indices:
            if iou_matrix[t, d] < self.iou_threshold:
                unmatched_tracks.add(t)
                unmatched_dets.add(d)
            else:
                self.tracks[t].update(detections[d]['bbox'])
                unmatched_tracks.discard(t)
                unmatched_dets.discard(d)
        
        for d in unmatched_dets:
            det = detections[d]
            self.tracks.append(Track(det['bbox'], det['class_id'], det['class_name']))
        
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        return self._get_active_tracks()
    
    def _get_active_tracks(self) -> List[dict]:
        active = []
        for track in self.tracks:
            if track.time_since_update < 1 and track.hits >= self.min_hits:
                bbox = track._z_to_bbox(track.kf.x[:4])
                active.append({
                    'track_id': track.id,
                    'bbox': [int(x) for x in bbox],
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'hits': track.hits
                })
        return active
