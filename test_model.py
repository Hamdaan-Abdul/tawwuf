"""Compare the published fine-tuned detector with pretrained YOLOv8n."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_MODEL_REPO = "Ultralytics/YOLOv8"
BASE_MODEL_FILE = "yolov8n.pt"
FINE_TUNED_MODEL_REPO = "Tawuff/tawwuf-traffic-yolov8n"
SAMPLE_VIDEO_REPO = "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages"
SAMPLE_VIDEO_FILE = "cartraffic04.mp4"
VEHICLE_LABELS = {"car", "bus", "truck", "van"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        help="MP4 to evaluate; defaults to a public Hugging Face traffic sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New output directory; defaults to runs/model-comparison-TIMESTAMP.",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def _vehicle_class_ids(model: Any) -> list[int]:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_ids = [
        int(class_id)
        for class_id, name in items
        if str(name).strip().lower() in VEHICLE_LABELS
    ]
    if not any(
        str(name).strip().lower() == "car"
        for name in (names.values() if isinstance(names, dict) else names)
    ):
        raise ValueError("The loaded model does not have a car class.")
    return class_ids


def _evaluate_model(
    label: str,
    weights: str,
    video: Path,
    output_dir: Path,
    stride: int,
    confidence: float,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(weights)
    class_ids = _vehicle_class_ids(model)
    frame_counts: list[int] = []
    class_totals: Counter[str] = Counter()
    annotated_video_directory = output_dir / label
    results = model.predict(
        source=str(video),
        classes=class_ids,
        conf=confidence,
        vid_stride=stride,
        stream=True,
        save=True,
        project=str(output_dir.resolve()),
        name=label,
        exist_ok=False,
        verbose=False,
    )
    for result in results:
        counts: Counter[str] = Counter()
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                counts[str(result.names[class_id]).strip().lower()] += 1
        frame_counts.append(sum(counts.values()))
        class_totals.update(counts)
        annotated_video_directory = Path(result.save_dir)

    if not frame_counts:
        raise ValueError(f"{label} produced no readable video frames.")
    return {
        "model": label,
        "frames_analyzed": len(frame_counts),
        "vehicle_detections": sum(frame_counts),
        "average_vehicles_per_sampled_frame": round(
            sum(frame_counts) / len(frame_counts), 2
        ),
        "peak_vehicles_in_sampled_frame": max(frame_counts),
        "detections_by_class": dict(class_totals),
        "annotated_video_directory": str(annotated_video_directory),
    }


def main() -> None:
    args = _parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")

    from huggingface_hub import hf_hub_download

    video = args.video or Path(
        hf_hub_download(
            repo_id=SAMPLE_VIDEO_REPO,
            repo_type="dataset",
            filename=SAMPLE_VIDEO_FILE,
        )
    )
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    output_dir = args.output or (
        Path("runs")
        / f"model-comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    pretrained_weights = hf_hub_download(
        repo_id=BASE_MODEL_REPO,
        filename=BASE_MODEL_FILE,
    )
    fine_tuned_weights = hf_hub_download(
        repo_id=FINE_TUNED_MODEL_REPO,
        filename="best.pt",
    )

    summaries = [
        _evaluate_model(
            "pretrained-yolov8n",
            pretrained_weights,
            video,
            output_dir,
            args.stride,
            args.confidence,
        ),
        _evaluate_model(
            "tawwuf-finetuned-yolov8n",
            fine_tuned_weights,
            video,
            output_dir,
            args.stride,
            args.confidence,
        ),
    ]
    report = {
        "video": str(video),
        "stride": args.stride,
        "confidence": args.confidence,
        "results": summaries,
        "warning": (
            "This sample comparison has no ground-truth labels. Detection "
            "counts alone do not establish which model is more accurate."
        ),
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
