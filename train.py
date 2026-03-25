from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=r"F:\7th sem lab\4709_HELMET_YOLOV8\images\helmet_dataset\data.yaml",
        epochs=50,
        imgsz=416,
        batch=2,
        device="cpu",
        project="runs/train",   
        name="helmet_yolov8",
        pretrained=True,
        patience=10,
        workers=0,
        cache=False,
        verbose=True
    )

    print("Training complete.")
    print(results)

if __name__ == "__main__":
    main()