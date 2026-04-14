from ultralytics import YOLO

BASE = r"C:\Users\shash\OneDrive\Desktop\clg_project\Industriguard-AI\training"

if __name__ == '__main__':
    model = YOLO(rf"{BASE}\ppe_model.pt")

    model.train(
        data=rf"{BASE}\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=rf"{BASE}\runs",
        name="yolo11s_ppe_v3",
        patience=30,
        save=True,
        plots=True,
        optimizer="AdamW",
        lr0=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        fliplr=0.5,
        mixup=0.2,
        copy_paste=0.2,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=4,   # explicitly set workers — avoids secondary multiprocessing issues on Windows
    )

    print("✅ Training complete!")
    print(rf"Best weights: {BASE}\runs\yolo11s_ppe_v3\weights\best.pt")