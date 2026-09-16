from ultralytics import YOLO

# Load our trained Apollo sponsor detector
model = YOLO(
    "runs/detect/models/apollo_brand_detector_v1-3/weights/best.pt"
)

# Test it on a frame the model did NOT see during training
results = model(
    "data/frames/match_01_0130s.jpg",
    conf=0.10
)

# Save the result with detection boxes
results[0].save(
    filename="outputs/detections/match_01_0130s_apollo_v1.jpg"
)

print("Apollo unseen-frame test complete!")