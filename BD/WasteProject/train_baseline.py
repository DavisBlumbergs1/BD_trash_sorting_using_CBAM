from ultralytics import YOLO
import torch
import os

def train_baseline_vanilla():
    print("\n=== STARTING BASELINE: VANILLA YOLO26 NANO FROM SCRATCH ===")
    
    # 1. Device selection
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training starting on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 2. LOAD VANILLA ARCHITECTURE BLUEPRINT (Added explicit task configuration)
    architecture_yaml = r"C:\Users\davis\Desktop\bakalaurs\practical\WasteProject\yolo26_baseline.yaml"
    model = YOLO(architecture_yaml, task="detect")

    # 3. Path Configuration
    yaml_path = r"C:\Users\davis\Desktop\bakalaurs\practical\WasteProject\datasets\waste_data_4\data.yaml"
    project_dir = r"C:\Users\davis\Desktop\bakalaurs\practical\runs"

    # 4. Training Execution
    model.train(
        data=yaml_path,
        epochs=40,           
        imgsz=640,          
        device=device,
        batch=8,            
        workers=4,          
        project=project_dir,
        name="waste_v26_baseline",
        optimizer='MuSGD',   
        amp=True,           
        save_period=10,      
    )

if __name__ == '__main__':
    train_baseline_vanilla()