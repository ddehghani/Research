from ultralytics import YOLO

model = YOLO("yolo11x_oiv7_new.pt") # or yaml if no pretrain

# Train the model
results = model.train(
    data="/home/ubuntu/openImages/Research/old_files/open.yaml",  # dataset config file
    epochs=30,
    imgsz=640,
)

# Save the trained weights
model.save("yolo11x_oiv7_newer.pt")
