from ultralytics import YOLO

# Edge
model = YOLO("yolov5n.yaml")  # this loads the model architecture only, no pretrained weights
results = model.train(
    data="VOC.yaml",
    epochs=20,
    imgsz=640,
)

model.save("voc_edge.pt")


# # Cloud
# model = YOLO("yolo11x.pt") 

# results = model.train(
#     data="VOC.yaml",
#     epochs=30,
#     imgsz=640,
# )

# model.save("voc_cloud.pt")