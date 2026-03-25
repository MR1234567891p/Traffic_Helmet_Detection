from ultralytics import YOLO

MODEL_PATH = r"F:\7th sem lab\4709_HELMET_YOLOV8\runs\detect\runs\train\helmet_yolov83\weights\best.pt"
DATA_YAML = r"F:\7th sem lab\4709_HELMET_YOLOV8\images\helmet_dataset\data.yaml"


def main():
    print("Loading trained model...")
    model = YOLO(MODEL_PATH)

    print("Running evaluation on test dataset...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",      
        imgsz=416,
        batch=2,
        device="cpu",
        project="runs/test",
        name="helmet_test_results"
    )

    print("\n========== TEST RESULTS ==========")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()