from ultralytics import YOLO

# Edge
model = YOLO("yolov5n.yaml")  # this loads the model architecture only, no pretrained weights

# Train
results = model.train(
    data="VOC.yaml",
    epochs=10, # fewer epochs
    imgsz=640,
)

# Save the intentionally bad model
model.save("yolo_voc_edge.pt")


# Cloud
model = YOLO("yolo11x.pt") 

# Train the model with bad settings
results = model.train(
    data="VOC.yaml",
    epochs=30,
    imgsz=640,
)

model.save("yolo_voc_cloud.pt")