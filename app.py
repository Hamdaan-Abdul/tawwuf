"""Hugging Face Space for traffic-video analysis and scenario exploration."""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from gradio import Progress

PROGRESS_TRACKER = Progress(track_tqdm=False)

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7
MODEL_REPO = "Ultralytics/YOLOv8"
MODEL_FILE = "yolov8n.pt"
TRAFFIC_MODEL_REPO = os.getenv("TRAFFIC_MODEL_REPO")
WEATHER_MODEL_REPO = "dima806/weather_types_image_detection"
LANE_SCENE_MODEL_REPO = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
WEATHER_RAIN_CONFIDENCE = 0.60
SAMPLE_VIDEO_REPO = "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages"
SAMPLE_VIDEOS = {
    "Car traffic sample 03 (13 MB)": "cartraffic03.mp4",
    "Car traffic sample 04 (9 MB)": "cartraffic04.mp4",
    "Crowd sample 03 (2.5 MB)": "crowd03.mp4",
    "Crowd sample 05 (4.1 MB)": "crowd05.mp4",
}


def recommended_speed_kmh(
    car_count: int,
    lane_count: int = 1,
    rain_detected: bool = False,
) -> int:
    """Return a demo speed based on detected cars per lane and estimated rain."""
    lane_count = _validate_lane_count(lane_count)

    cars_per_lane = car_count / lane_count
    if cars_per_lane < 5:
        speed = 100
    elif cars_per_lane <= 15:
        speed = 80
    else:
        speed = 60

    if rain_detected:
        speed = max(60, speed - 20)
    return speed


def _validate_lane_count(lane_count: Any) -> int:
    """Require a positive whole number of lanes."""
    try:
        normalized = int(lane_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Lane count must be a whole number of at least one.") from error
    if normalized < 1 or normalized != lane_count:
        raise ValueError("Lane count must be a whole number of at least one.")
    return normalized


def _estimate_visible_lane_count(frame: Any, road_mask: Any = None) -> int | None:
    """Estimate lane count from road-surface line geometry in one frame."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    roi = np.zeros_like(edges)
    cv2.fillPoly(
        roi,
        [
            np.array(
                [
                    (0, int(height * 0.48)),
                    (width, int(height * 0.48)),
                    (width, height - 1),
                    (0, height - 1),
                ],
                dtype=np.int32,
            )
        ],
        255,
    )
    analysis_region = roi
    if road_mask is not None:
        road_mask = np.asarray(road_mask, dtype=np.uint8)
        if road_mask.shape != edges.shape:
            raise ValueError("Road segmentation mask must match the video frame.")
        road_mask = cv2.dilate(
            road_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        analysis_region = cv2.bitwise_and(analysis_region, road_mask)

    lines = cv2.HoughLinesP(
        cv2.bitwise_and(edges, analysis_region),
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(24, int(height * 0.045)),
        maxLineGap=max(20, int(height * 0.08)),
    )
    if lines is None:
        return None

    bottom_y = height * 0.94
    candidates: list[tuple[float, float]] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (float(value) for value in line)
        delta_y = y2 - y1
        if abs(delta_y) < height * 0.04:
            continue
        slope = (x2 - x1) / delta_y
        if abs(slope) < 0.06 or abs(slope) > 6.0:
            continue
        projected_x = x1 + slope * (bottom_y - y1)
        if 0.01 * width <= projected_x <= 0.99 * width:
            candidates.append((projected_x, math.hypot(x2 - x1, delta_y)))

    if len(candidates) < 4:
        return None

    candidates.sort()
    clusters: list[list[tuple[float, float]]] = []
    cluster_tolerance = max(12.0, width * 0.025)
    for candidate in candidates:
        if (
            not clusters
            or candidate[0] - clusters[-1][-1][0] > cluster_tolerance
        ):
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    supported_markings = [
        cluster
        for cluster in clusters
        if sum(length for _, length in cluster) >= height * 0.08
    ]
    # Two carriageways have two more outer boundaries than lane intervals.
    estimated_lanes = len(supported_markings) - 2
    if not 1 <= estimated_lanes <= 12:
        return None
    return estimated_lanes


@lru_cache(maxsize=1)
def load_lane_scene_segmenter() -> tuple[Any, Any]:
    """Load a Cityscapes road/sidewalk segmenter from the Hugging Face Hub."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(LANE_SCENE_MODEL_REPO)
    model = SegformerForSemanticSegmentation.from_pretrained(
        LANE_SCENE_MODEL_REPO
    ).eval()
    return processor, model


def _segment_road_surface(frame: Any) -> Any:
    """Return a road-only mask, excluding Cityscapes sidewalk predictions."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as torch_functional

    processor, model = load_lane_scene_segmenter()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        labels = torch_functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)[0]
    return (labels.cpu().numpy() == 0).astype(np.uint8) * 255


def _select_consistent_lane_estimate(estimates: list[int]) -> int | None:
    """Choose a conservative estimate or reject inconsistent multi-frame results."""
    if len(estimates) < 2:
        return None
    if len(estimates) == 2:
        return min(estimates)
    if max(estimates) - min(estimates) > 1:
        return None

    from collections import Counter

    counts = Counter(estimates)
    max_count = max(counts.values())
    return min(value for value, count in counts.items() if count == max_count)


def _detection_class_ids(model: Any) -> tuple[list[int], set[int], set[int], set[int]]:
    """Resolve target class IDs by name for COCO and fine-tuned model heads."""
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_groups: dict[str, set[int]] = {
        "person": set(),
        "car": set(),
        "heavy": set(),
    }
    for class_id, class_name in items:
        normalized = str(class_name).strip().lower()
        if normalized == "person":
            class_groups["person"].add(int(class_id))
        elif normalized in {"car", "cars"}:
            class_groups["car"].add(int(class_id))
        elif normalized in {"truck", "trucks", "bus", "van"}:
            class_groups["heavy"].add(int(class_id))

    if not class_groups["car"]:
        raise RuntimeError("The loaded detector has no car class.")
    target_ids = sorted(
        class_groups["person"] | class_groups["car"] | class_groups["heavy"]
    )
    return (
        target_ids,
        class_groups["person"],
        class_groups["car"],
        class_groups["heavy"],
    )


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load fine-tuned weights when configured, otherwise the pretrained model."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    if TRAFFIC_MODEL_REPO:
        weights_path = hf_hub_download(
            repo_id=TRAFFIC_MODEL_REPO,
            filename="best.pt",
        )
        return YOLO(weights_path)

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(weights_path)


@lru_cache(maxsize=1)
def load_weather_classifier() -> Any:
    """Load an image weather classifier from the Hugging Face Hub."""
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=WEATHER_MODEL_REPO,
        device=-1,
    )


def _classify_weather_frame(frame: Any) -> dict:
    """Classify one OpenCV BGR frame with the cached Hugging Face model."""
    import cv2
    from PIL import Image

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = load_weather_classifier()(Image.fromarray(rgb_frame), top_k=1)
    if predictions and isinstance(predictions[0], list):
        prediction = predictions[0][0]
    elif predictions:
        prediction = predictions[0]
    else:
        raise RuntimeError("The weather classifier returned no predictions.")

    return {
        "label": str(prediction["label"]),
        "score": float(prediction["score"]),
    }


def _summarize_weather(predictions: list[dict]) -> tuple[str, float, bool]:
    """Aggregate sampled weather classifications and gate the rain adjustment."""
    if not predictions:
        return "Unavailable", 0.0, False

    from collections import Counter

    labels = [prediction["label"].strip().lower() for prediction in predictions]
    label_counts = Counter(labels)
    label = label_counts.most_common(1)[0][0]
    confidence = sum(
        prediction["score"]
        for prediction, predicted_label in zip(predictions, labels)
        if predicted_label == label
    ) / label_counts[label]
    rain_detected = (
        label == "rain"
        and label_counts[label] > len(predictions) / 2
        and confidence >= WEATHER_RAIN_CONFIDENCE
    )
    return label.title(), confidence, rain_detected


def load_sample_video(sample_name: str) -> str:
    """Download a selected public sample clip from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    filename = SAMPLE_VIDEOS.get(sample_name)
    if filename is None:
        raise ValueError("Choose one of the listed sample videos.")

    return hf_hub_download(
        repo_id=SAMPLE_VIDEO_REPO,
        repo_type="dataset",
        filename=filename,
    )


def _context_summary(
    school_peak: bool,
    road_condition: str,
    high_risk_zone: bool,
) -> str:
    """Describe user-selected scenario context without implying sensor data."""
    flags = [
        "School-peak scenario: {}".format("On" if school_peak else "Off"),
        f"Road scenario: {road_condition}",
        "High-risk-zone scenario: {}".format("On" if high_risk_zone else "Off"),
    ]
    return " · ".join(flags)


def _make_trend_plot(timeline: Any) -> Any:
    """Create a static detection trend chart for Gradio's Matplotlib plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4))
    for column, label in (
        ("Cars", "Cars"),
        ("Trucks", "Trucks"),
        ("People detected", "People"),
    ):
        axis.plot(timeline["Time (seconds)"], timeline[column], label=label)

    axis.set_xlabel("Time in clip (seconds)")
    axis.set_ylabel("Detections per analyzed frame")
    axis.set_title("Vehicle and pedestrian detections")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _resolve_video_path(video_value: Any) -> str:
    """Extract the local file path from Gradio Video input formats."""
    if isinstance(video_value, dict):
        video_value = video_value.get("video")
    if isinstance(video_value, dict):
        video_value = video_value.get("path")
    if not video_value:
        raise ValueError("Upload or load a traffic video first.")
    return str(video_value)


def process_video(
    video_path: Any,
    lane_count: int = 1,
    detect_lanes_automatically: bool = False,
    frame_stride: int = 1,
    school_peak: bool = False,
    road_condition: str = "Normal",
    high_risk_zone: bool = False,
    progress: Progress = PROGRESS_TRACKER,
) -> tuple[str, str, str, Any, str]:
    """Annotate a video and return recommendation, trend data, and summary."""
    lane_count = _validate_lane_count(lane_count)
    if frame_stride < 1:
        raise ValueError("Frame interval must be at least one.")
    frame_stride = int(frame_stride)
    video_path = _resolve_video_path(video_path)

    import cv2
    import pandas as pd

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    lane_estimate_frame_indices = (
        {
            0,
            total_frames // 4,
            total_frames // 2,
            (3 * total_frames) // 4,
            total_frames - 1,
        }
        if total_frames > 0
        else {0}
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = Path(output_file.name)

    writer = None
    frame_number = 0
    sample_count = 0
    tracked_car_ids: set[int] = set()
    tracked_truck_ids: set[int] = set()
    tracking_ids_available = True
    lane_count_source = "manual"
    lane_estimate_status = ""
    lane_estimates: list[int] = []
    weather_predictions = []
    weather_frame_indices = (
        {0, total_frames // 2, total_frames - 1} if total_frames > 0 else {0}
    )
    records: list[dict] = []
    succeeded = False

    try:
        if detect_lanes_automatically:
            progress(
                0,
                desc="Loading road/sidewalk segmentation model from Hugging Face...",
            )
            load_lane_scene_segmenter()
        progress(0, desc="Loading YOLOv8 from Hugging Face...")
        model = load_model()
        (
            detection_class_ids,
            person_class_ids,
            car_class_ids,
            heavy_vehicle_class_ids,
        ) = _detection_class_ids(model)
        progress(0.03, desc="Loading weather classifier from Hugging Face...")
        load_weather_classifier()
        progress(0.05, desc="Analyzing video frames...")
        progress_interval = max(1, total_frames // 50) if total_frames > 0 else 30

        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if (
                detect_lanes_automatically
                and frame_number in lane_estimate_frame_indices
            ):
                road_mask = _segment_road_surface(frame)
                detected_lane_count = _estimate_visible_lane_count(
                    frame,
                    road_mask=road_mask,
                )
                if detected_lane_count is not None:
                    lane_estimates.append(detected_lane_count)

            if frame_number in weather_frame_indices:
                weather_predictions.append(_classify_weather_frame(frame))

            if frame_number % frame_stride == 0:
                result = model.track(
                    source=frame,
                    classes=detection_class_ids,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]
                boxes = result.boxes
                class_ids = boxes.cls.tolist() if boxes is not None else []
                people = sum(int(class_id) in person_class_ids for class_id in class_ids)
                cars = sum(int(class_id) in car_class_ids for class_id in class_ids)
                trucks = sum(
                    int(class_id) == TRUCK_CLASS_ID for class_id in class_ids
                )
                if boxes is not None and len(boxes):
                    track_ids = boxes.id
                    if track_ids is None:
                        tracking_ids_available = False
                    else:
                        for class_id, track_id in zip(
                            class_ids, track_ids.int().tolist()
                        ):
                            if int(class_id) in car_class_ids:
                                tracked_car_ids.add(track_id)
                            elif int(class_id) in heavy_vehicle_class_ids:
                                tracked_truck_ids.add(track_id)

                if boxes is not None and len(boxes):
                    box_coords = boxes.xyxy.tolist()
                    box_area = sum(
                        max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        for x1, y1, x2, y2 in box_coords
                    )
                    frame_area = float(frame.shape[0] * frame.shape[1])
                    occupancy = min(100.0, box_area / frame_area * 100.0)
                else:
                    occupancy = 0.0

                records.append(
                    {
                        "Time (seconds)": frame_number / fps,
                        "Cars": cars,
                        "Trucks": trucks,
                        "People detected": people,
                        "Detected box area (%)": round(occupancy, 2),
                    }
                )
                annotated_frame = result.plot()
                sample_count += 1
            else:
                annotated_frame = frame

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"avc1"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create an H.264 processed video. "
                        "Check that OpenCV has an H.264 encoder available."
                    )

            writer.write(annotated_frame)
            frame_number += 1
            if frame_number % progress_interval == 0:
                if total_frames > 0:
                    fraction = min(frame_number / total_frames, 1.0)
                    progress(
                        0.05 + 0.9 * fraction,
                        desc=f"Analyzing frame {frame_number} of {total_frames}",
                    )
                else:
                    progress(
                        None,
                        desc=f"Analyzed {frame_number} frames",
                    )

        if not frame_number or not records:
            raise ValueError("The uploaded video contains no readable frames.")

        selected_lane_estimate = _select_consistent_lane_estimate(lane_estimates)
        if selected_lane_estimate is not None:
            lane_count = selected_lane_estimate
            lane_count_source = (
                f"automatic road-area estimate from {len(lane_estimates)} frame(s)"
            )
        elif detect_lanes_automatically:
            lane_estimate_status = (
                f"Automatic road-area estimates {lane_estimates or 'unavailable'} "
                "were inconclusive; the entered lane count was used."
            )

        timeline = pd.DataFrame(records)
        progress(0.97, desc="Preparing charts and results...")
        trend_plot = _make_trend_plot(timeline)
        succeeded = True

        weather_label, weather_confidence, rain_detected = _summarize_weather(
            weather_predictions
        )
        traffic_peak = timeline.loc[timeline["Cars"].idxmax()]
        peak_car_count = int(traffic_peak["Cars"])
        cars_per_lane = peak_car_count / lane_count
        speed = recommended_speed_kmh(
            peak_car_count,
            lane_count=lane_count,
            rain_detected=rain_detected,
        )
        if tracking_ids_available:
            unique_vehicle_summary = (
                f"Unique tracked vehicles in clip: {len(tracked_car_ids)} cars, "
                f"{len(tracked_truck_ids)} trucks/buses/vans"
            )
        else:
            unique_vehicle_summary = (
                "Unique vehicle count unavailable: the model did not return "
                "tracking IDs for every detected vehicle."
            )
        weather_text = (
            f"**Video weather estimate:** {weather_label} "
            f"({weather_confidence:.0%} confidence, "
            f"{len(weather_predictions)} sampled frame(s))"
        )
        if rain_detected:
            weather_text += (
                "\n\nRain confidence passed the demo threshold; the speed "
                "output was lowered one band."
            )
        else:
            weather_text += (
                "\n\nRain adjustment not applied; weather estimates are "
                "experimental and may be incorrect."
            )
        speed_text = (
            f"## {speed} km/h\n"
            f"Demo only · Peak sampled frame: {peak_car_count} cars across "
            f"{lane_count} lane(s) ({cars_per_lane:.1f} cars per lane)"
        )
        speed_text += f"\n\nLane count: {lane_count} ({lane_count_source})"
        if lane_estimate_status:
            speed_text += f"\n\n{lane_estimate_status}"
        speed_text += f"\n\n{unique_vehicle_summary}"

        traffic_lull = timeline.loc[timeline["Cars"].idxmin()]
        people_peak = timeline.loc[timeline["People detected"].idxmax()]
        crowd_note = (
            "People detected are a rough visual count, not a measure of "
            "pedestrian behavior or unique people."
        )
        timing_note = (
            "Peak times below are elapsed times within this uploaded clip; "
            "they do not establish daily rush-hour or school-hour patterns."
        )
        source_frames = total_frames if total_frames > 0 else frame_number
        lane_source_summary = f"- Lane count source: {lane_count_source}"
        if lane_estimates:
            lane_source_summary += f" · estimates: {lane_estimates}"
        if lane_estimate_status:
            lane_source_summary += f" · {lane_estimate_status}"
        if person_class_ids:
            people_summary = (
                f"- Peak people detections: "
                f"{int(people_peak['People detected'])} at "
                f"{float(people_peak['Time (seconds)']):.1f}s into the clip\n"
            )
        else:
            people_summary = (
                "- People detection: unavailable in this vehicle-only model\n"
            )
        summary = (
            "### Clip analysis\n"
            f"- Frames: {source_frames} · analyzed: {sample_count} · "
            f"interval: every {frame_stride} frame(s)\n"
            f"- Peak car detections in one analyzed frame: {peak_car_count} at "
            f"{float(traffic_peak['Time (seconds)']):.1f}s into the clip\n"
            f"- Lowest car detections: {int(traffic_lull['Cars'])} at "
            f"{float(traffic_lull['Time (seconds)']):.1f}s into the clip\n"
            f"{people_summary}"
            f"- Peak detected box area: "
            f"{float(timeline['Detected box area (%)'].max()):.1f}% of frame\n\n"
            f"- Lanes used: {lane_count} · peak sampled cars per lane: "
            f"{cars_per_lane:.1f}\n\n"
            f"{lane_source_summary}\n"
            f"- {unique_vehicle_summary}\n"
            "  Counts are approximate: tracking only sees analyzed frames and "
            "can miss or duplicate vehicles, especially at larger intervals. "
            "Lower the interval to improve tracking continuity.\n\n"
            "**Other scenario inputs (simulated, not live feeds):** "
            f"{_context_summary(school_peak, road_condition, high_risk_zone)}\n\n"
            f"**Scope:** {crowd_note} {timing_note}\n\n"
            "**Demo rule:** thresholds apply to cars per lane "
            "(<5: 100; 5–15: 80; >15: 60 km/h). If rain wins a majority of "
            "sampled weather predictions at ≥60% mean confidence, lower one "
            "band (100→80, 80→60, 60 stays 60). These values are not validated "
            "speed limits. "
            "No live camera, weather service, school calendar, road-condition "
            "feed, accident data, or traffic sign is connected."
        )
        progress(1, desc="Analysis complete")
        return str(output_path), speed_text, weather_text, trend_plot, summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not succeeded:
            output_path.unlink(missing_ok=True)


def search_hf_datasets(query: str) -> Any:
    """Search the Hugging Face Hub dataset catalog and report license metadata."""
    if not query.strip():
        raise ValueError("Enter a topic to search.")

    import pandas as pd
    from huggingface_hub import list_datasets

    rows = []
    for dataset in list_datasets(
        search=query.strip(),
        sort="downloads",
        direction=-1,
        limit=10,
    ):
        card = getattr(dataset, "cardData", None)
        license_name = None
        if card is not None:
            license_name = (
                card.get("license")
                if isinstance(card, dict)
                else getattr(card, "license", None)
            )
        rows.append(
            {
                "Dataset": dataset.id,
                "License in Hub card": license_name or "Not declared",
                "Downloads": getattr(dataset, "downloads", 0) or 0,
                "Hugging Face page": f"https://huggingface.co/datasets/{dataset.id}",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Dataset",
            "License in Hub card",
            "Downloads",
            "Hugging Face page",
        ],
    )


def launch_app() -> None:
    import gradio as gr

    with gr.Blocks(
        title="Tawwuf | Traffic Intelligence Demo",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
        css=(
            ".speed-card { text-align: center; padding: 1.5rem; "
            "border-radius: 1rem; background: linear-gradient(135deg, "
            "#0b1f33, #145374); color: white; }"
            ".speed-card h2 { font-size: 3.5rem; margin: .2rem; }"
            ".safety-note { border-left: 4px solid #e6a700; "
            "padding: .8rem 1rem; background: #fff8e6; }"
        ),
    ) as demo:
        gr.Markdown(
            "# Tawwuf Traffic Intelligence\n"
            "### A Hugging Face-powered research prototype for Doha traffic scenarios\n\n"
            "Upload an authorized clip to explore vehicle and pedestrian-count "
            "trends. Weather is estimated from sampled frames; school, road, "
            "and zone inputs remain simulated until approved local data feeds "
            "are available."
        )
        gr.Markdown(
            "<div class='safety-note'><b>Research demo only.</b> This app "
            "does not connect to government cameras, detect accidents, or "
            "control road signs. Its speed output is not suitable for "
            "operational use.</div>"
        )

        with gr.Column():
            with gr.Accordion("Analyze a video", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_video = gr.Video(
                            label="Traffic video",
                            sources=["upload"],
                            format="mp4",
                        )
                        sample_video = gr.Dropdown(
                            choices=list(SAMPLE_VIDEOS),
                            value="Crowd sample 03 (2.5 MB)",
                            label="Or choose a repeatable Hugging Face sample",
                            info=(
                                "The selected clip downloads from the public "
                                "dataset the first time, then uses the local cache."
                            ),
                        )
                        load_sample_button = gr.Button("Load selected sample")
                        gr.Markdown(
                            "Sample source: "
                            "[Small Crowd and Traffic Video Footages]"
                            "(https://huggingface.co/datasets/"
                            "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages) "
                            "— MIT license declared in Hub metadata. The dataset "
                            "card is sparse; verify source terms before reuse "
                            "outside testing."
                        )
                        frame_stride = gr.Slider(
                            minimum=1,
                            maximum=15,
                            value=15,
                            step=1,
                            label="Analyze every Nth frame",
                            info=(
                                "Default 15 analyzes fewer frames and is faster; "
                                "lower values give denser sampling and improve "
                                "tracking continuity. Unsampled frames are not "
                                "boxed."
                            ),
                        )
                        lane_count = gr.Number(
                            value=3,
                            minimum=1,
                            maximum=12,
                            precision=0,
                            label="Number of visible road lanes",
                            info=(
                                "Enter the lanes in the camera view. Counts "
                                "are divided by this value unless automatic "
                                "estimation succeeds."
                            ),
                        )
                        detect_lanes_automatically = gr.Checkbox(
                            label="Estimate visible lane count automatically",
                            value=False,
                            info=(
                                "Experimental Hugging Face road/sidewalk "
                                "segmentation plus line estimates across up to "
                                "five points. Two estimates use the lower count; "
                                "highly inconsistent results use the manual count."
                            ),
                        )
                        with gr.Accordion("Scenario controls (simulated)", open=True):
                            school_peak = gr.Checkbox(
                                label="School arrival / dismissal scenario",
                                value=False,
                            )
                            road_condition = gr.Dropdown(
                                [
                                    "Normal",
                                    "Wet road",
                                    "Roadworks",
                                    "Reduced visibility",
                                ],
                                value="Normal",
                                label="Road condition",
                            )
                            high_risk_zone = gr.Checkbox(
                                label="Known high-risk-zone scenario",
                                value=False,
                            )
                        load_sample_button.click(
                            fn=load_sample_video,
                            inputs=sample_video,
                            outputs=input_video,
                        )
                        analyze_button = gr.Button("Analyze clip", variant="primary")
                    with gr.Column(scale=1):
                        speed_output = gr.Markdown(
                            "## -- km/h\nDemo recommendation",
                            elem_classes=["speed-card"],
                        )
                        weather_output = gr.Markdown(
                            "**Video weather estimate:** waiting for analysis\n\n"
                            "Experimental frame classifier; not validated for "
                            "traffic-camera weather or speed control."
                        )
                        output_video = gr.Video(
                            label="Video with sampled-frame detections"
                        )
                trend_plot = gr.Plot(
                    label="Traffic and people detected over clip time",
                )
                analysis_summary = gr.Markdown()

                analyze_button.click(
                    fn=process_video,
                    inputs=[
                        input_video,
                        lane_count,
                        detect_lanes_automatically,
                        frame_stride,
                        school_peak,
                        road_condition,
                        high_risk_zone,
                    ],
                    outputs=[
                        output_video,
                        speed_output,
                        weather_output,
                        trend_plot,
                        analysis_summary,
                    ],
                    concurrency_limit=1,
                )

            with gr.Accordion("Explore HF datasets", open=False):
                gr.Markdown(
                    "Search the Hugging Face dataset catalog for research "
                    "candidates. Search results are not automatically used "
                    "for Qatar: review the dataset card, geographic coverage, "
                    "license, privacy terms, and data quality before use."
                )
                dataset_query = gr.Dropdown(
                    [
                        "traffic video",
                        "traffic weather",
                        "pedestrian traffic",
                        "road accident",
                    ],
                    value="traffic video",
                    label="Research topic",
                    allow_custom_value=True,
                )
                search_button = gr.Button("Search Hugging Face datasets")
                dataset_results = gr.Dataframe(
                    headers=[
                        "Dataset",
                        "License in Hub card",
                        "Downloads",
                        "Hugging Face page",
                    ],
                    datatype=["str", "str", "number", "str"],
                    interactive=False,
                    label="Dataset candidates",
                )
                search_button.click(
                    fn=search_hf_datasets,
                    inputs=dataset_query,
                    outputs=dataset_results,
                )
                gr.Markdown(
                    "**Data note:** Public datasets can help evaluate general "
                    "vehicle detection, but they do not supply authoritative "
                    "Qatar school hours, road conditions, accident-prone "
                    "locations, or citywide peak-hour patterns."
                )

            with gr.Accordion("How to interpret this demo", open=False):
                gr.Markdown(
                    "### What the prototype measures\n"
                    "- YOLOv8 detects cars, trucks, and people in sampled "
                    "frames; ByteTrack estimates unique cars and trucks "
                    "across those frames.\n"
                    "- The trend plot shows detections during this clip only. "
                    "It does not track unique people or infer crossing intent "
                    "or other pedestrian behavior.\n"
                    "- Detected box area is a rough image-occupancy proxy, "
                    "not calibrated road density.\n"
                    "- The 100/80/60 km/h thresholds are the original demo "
                    "rule, based on car detections in the busiest analyzed "
                    "frame. Scenario controls do not alter it. Tracking totals "
                    "are approximate, especially with larger frame intervals.\n\n"
                    "### What is needed before a Qatar pilot\n"
                    "Authorized camera access; privacy review; reliable "
                    "timestamped weather, school-calendar, roadworks, and "
                    "incident feeds; locally representative labeled data; "
                    "traffic-engineering validation; human approval; and "
                    "fail-safe sign-control integration. This Space provides "
                    "none of those operational connections."
                )

    demo.launch(share=True)


if __name__ == "__main__":
    launch_app()
