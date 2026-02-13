from ultralytics import YOLO
import os

def download_yolo_models():
    models_dir = "./models/weights"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading YOLOv8 models...")
    
    # Detection models
    detection_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    for variant in detection_models:
        print(f"Downloading {variant}...")
        model = YOLO(variant)
        print(f"{variant} downloaded successfully")
    
    # Segmentation models
    print("\nDownloading YOLOv8 segmentation models...")
    seg_models = ['yolov8n-seg.pt', 'yolov8s-seg.pt']
    
    for variant in seg_models:
        print(f"Downloading {variant}...")
        model = YOLO(variant)
        print(f"{variant} downloaded successfully")

if __name__ == "__main__":
    download_yolo_models()
