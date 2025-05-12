from ultralytics import YOLO

# Load a pretrained model
model = YOLO("/home/devdem/MultiTask_vs_Yolo_Unet/yolov8n.pt")

# Train the model on Apple silicon chip (M1/M2/M3/M4)
results = model.train(data="nuimages.yaml", epochs=1, imgsz=640, device="0")