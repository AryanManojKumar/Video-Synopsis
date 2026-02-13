from ultralytics import YOLO
import os

def download_yolo_models():
    models_dir = "./models/weights"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading YOLOv8 models...")
    
    model_variants = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    for variant in model_variants:
        print(f"Downloading {variant}...")
        model = YOLO(variant)
        print(f"{variant} downloaded successfully")

if __name__ == "__main__":
    download_yolo_models()
