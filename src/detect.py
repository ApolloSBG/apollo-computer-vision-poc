# Install YOLO package / Object Detection Software
#Image Detection using YOLO
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run object detection on our extracted football frame
results = model("data/frames/match_01_frame_60s.jpg")

# Save an image showing the detected objects
results[0].save(filename="outputs/detections/match_01_60s_detected.jpg")

print("Detection complete!")

# Fine-tune YOLO on Apollo sponsor brands
model = YOLO("yolo11n.pt")

model.train(
    data="data/annotations/batch01/data.yaml",
    epochs=30,
    imgsz=640,
    project="models",
    name="apollo_brand_detector_v1"
)

# Test Apollo model on an unseen frame
apollo_model = YOLO(
    "runs/detect/models/apollo_brand_detector_v1-3/weights/best.pt"
)

test_results = apollo_model(
    "data/frames/match_01_0130s.jpg",
    conf=0.10
)

test_results[0].save(
    filename="outputs/detections/match_01_0130s_apollo_v1.jpg"
)

print("Apollo unseen-frame test complete!")