from ultralytics import YOLO

# 1. Load pretrained Nano model
model = YOLO("yolo11n.pt")

# 2. Train on your custom dataset
results = model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device="mps"
)

# 3. Load the BEST model produced during training
model = YOLO("runs/detect/train/weights/best.pt")

# 4. Test on a new image
results = model.predict(
    source="test.jpg",
    conf=0.4,
    save=True
)
print("Prediction for test.jpg (A Chamomille)", results)

# 5. Export the trained model for Raspberry Pi
model.export(format="ncnn")