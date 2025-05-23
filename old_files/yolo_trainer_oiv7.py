from ultralytics import YOLO

model = YOLO("yolov9c.pt") # or yaml if no pretrain

# Train the model
results = model.train(
    workers = 32,
    data="/home/ubuntu/openImages/Research/old_files/datasets/modified-oiv7-map3/dataset.yaml",  # dataset config file
    epochs=50,
    imgsz=640,
    patience=5,
    optimizer="auto",
    cache="ram",
)

# Save the trained weights
model.save("oiv7_cloud.pt")
