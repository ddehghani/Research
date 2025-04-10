from ultralytics import YOLO

# Load a model
model = YOLO("yolov5nu.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VOC.yaml", epochs=3, imgsz=640, batch=4, workers=2 )

# Evaluate the model
results = model.val()

# Save the model
model.save("yolov5nu_voc_trained_min.pt")  # save the model