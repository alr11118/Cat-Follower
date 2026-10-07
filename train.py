# Load Data Set
# Set up model
# 'Train' it????
# Test it???
# Export it to be used on the Raspberry Pi
from ultralytics import YOLO

# 1. Load pre-trained Nano model
model = YOLO("yolo11n.pt") 

# 2. Train on custom dataset
results = model.train(
    data="data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    device="mps" # Use 0 for Nvidia GPU, 'mps' for Apple Silicon, or 'cpu'
)

# 3. Export to NCNN format for maximum CPU speed
model.export(format="ncnn")
# how to text?