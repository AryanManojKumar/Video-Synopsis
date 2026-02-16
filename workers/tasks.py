from workers.celery_app import celery_app
from core.detector import ObjectDetector
from core.tracker import MultiObjectTracker
from core.background import BackgroundExtractor
from core.tubes import TubeGenerator
from core.optimizer import SpatialLaneOptimizer
from core.renderer import SynopsisRenderer
from utils.video import VideoProcessor
from utils.storage import upload_to_s3, download_from_s3
from db.models import Job
from db.session import get_db
import cv2

@celery_app.task(bind=True)
def process_video_synopsis(self, job_id: str, video_path: str, 
                          compression_ratio: float):
    db = next(get_db())
    job = db.query(Job).filter(Job.id == job_id).first()
    
    try:
        job.status = "processing"
        job.progress = 0.0
        db.commit()
        
        local_video = download_from_s3(video_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 10})
        job.progress = 10.0
        db.commit()
        
        video_proc = VideoProcessor(local_video)
        frames = video_proc.extract_frames()
        fps = video_proc.get_fps()
        
        self.update_state(state='PROGRESS', meta={'progress': 20})
        job.progress = 20.0
        db.commit()
        
        bg_extractor = BackgroundExtractor()
        background = bg_extractor.extract_from_frames(frames[::30])
        
        self.update_state(state='PROGRESS', meta={'progress': 30})
        job.progress = 30.0
        db.commit()
        
        detector = ObjectDetector()  # Reads use_segmentation from settings
        tracker = MultiObjectTracker()
        
        tracks_per_frame = []
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            tracks = tracker.update(detections, i)
            tracks_per_frame.append(tracks)
            
            if i % 100 == 0:
                progress = 30 + (i / len(frames)) * 30
                self.update_state(state='PROGRESS', meta={'progress': progress})
                job.progress = progress
                db.commit()
        
        self.update_state(state='PROGRESS', meta={'progress': 60})
        job.progress = 60.0
        db.commit()
        
        tube_gen = TubeGenerator()
        tubes = tube_gen.generate_tubes(tracks_per_frame, frames)
        
        self.update_state(state='PROGRESS', meta={'progress': 70})
        job.progress = 70.0
        db.commit()
        
        optimizer = SpatialLaneOptimizer()
        optimizer.compression_ratio = compression_ratio
        placements = optimizer.optimize(tubes, len(frames))
        
        self.update_state(state='PROGRESS', meta={'progress': 85})
        job.progress = 85.0
        db.commit()
        
        renderer = SynopsisRenderer(background, fps)
        output_path = f"/tmp/synopsis_{job_id}.mp4"
        renderer.render(placements, output_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 95})
        job.progress = 95.0
        db.commit()
        
        with open(output_path, 'rb') as f:
            result_url = upload_to_s3(f.read(), f"results/{job_id}/synopsis.mp4")
        
        job.status = "completed"
        job.progress = 100.0
        job.result_url = result_url
        db.commit()
        
        return {"status": "completed", "result_url": result_url}
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.commit()
        raise
