from ultralytics import YOLO

model = YOLO("/home/ubuntu/openImages/Research/old_files/runs/detect/train10/weights/best.pt")  # or any .pt model
results = model.val(data="/home/ubuntu/openImages/Research/old_files/datasets/modified-oiv7-map1/dataset.yaml")  # can also pass val images/labels dir

names = model.names  # class index to name mapping

for i, name in names.items():
    precision = results.box.p[i]
    recall = results.box.r[i]
    map50 = results.box.map50[i]
    map5095 = results.box.map[i]
    
    print(f"{name:<15} | Precision: {precision:.3f} | Recall: {recall:.3f} | "
          f"mAP@0.5: {map50:.3f} | mAP@0.5:0.95: {map5095:.3f}")

print("Overall Precision:", results.box.p.mean())
print("Overall Recall:", results.box.r.mean())
print("Overall mAP@0.5:", results.box.map50.mean())
print("Overall mAP@0.5:0.95:", results.box.map.mean())