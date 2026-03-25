from ultralytics import YOLO

def main():
    model = YOLO(r"F:\7th sem lab\4709_HELMET_YOLOV8\runs\detect\runs\train\helmet_yolov83\weights\best.pt")

    metrics = model.val(
        data=r"F:\7th sem lab\4709_HELMET_YOLOV8\images\helmet_dataset\data.yaml",
        split="test",
        imgsz=416,
        batch=2,
        device="cpu"
    )

    print("Evaluation complete.")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

if __name__ == "__main__":
    main()