"""Count cars and trucks in each frame of a video using YOLOv8."""

import argparse
from pathlib import Path


CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count cars and trucks in every frame of a video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video file does not exist: {args.video}")

    import cv2
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        parser.error(f"could not open video: {args.video}")
    capture.release()

    weights_path = hf_hub_download(
        repo_id="Ultralytics/YOLOv8",
        filename="yolov8n.pt",
    )
    model = YOLO(weights_path)

    print("frame,car_count,truck_count,total_count")
    for frame_number, result in enumerate(
        model.predict(
            source=str(args.video),
            stream=True,
            classes=[CAR_CLASS_ID, TRUCK_CLASS_ID],
            verbose=False,
        ),
        start=1,
    ):
        class_ids = (
            result.boxes.cls.tolist() if result.boxes is not None else []
        )
        car_count = sum(class_id == CAR_CLASS_ID for class_id in class_ids)
        truck_count = sum(class_id == TRUCK_CLASS_ID for class_id in class_ids)
        print(
            f"{frame_number},{car_count},{truck_count},"
            f"{car_count + truck_count}"
        )


if __name__ == "__main__":
    main()
