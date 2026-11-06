"""Hugging Face Space for traffic-video analysis and scenario exploration."""

from __future__ import annotations

import math
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7
MODEL_REPO = "Ultralytics/YOLOv8"
MODEL_FILE = "yolov8n.pt"
SAMPLE_VIDEO_REPO = "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages"
SAMPLE_VIDEOS = {
    "Car traffic sample 03 (13 MB)": "cartraffic03.mp4",
    "Car traffic sample 04 (9 MB)": "cartraffic04.mp4",
    "Crowd sample 03 (2.5 MB)": "crowd03.mp4",
    "Crowd sample 05 (4.1 MB)": "crowd05.mp4",
}


def recommended_speed_kmh(car_count: int) -> int:
    """Return the demo speed recommendation for a car count."""
    if car_count < 5:
        return 100
    if car_count <= 15:
        return 80
    return 60


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load YOLOv8n weights from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(weights_path)


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
    weather: str,
    school_peak: bool,
    road_condition: str,
    high_risk_zone: bool,
) -> str:
    """Describe user-selected scenario context without implying sensor data."""
    flags = [
        f"Weather: {weather}",
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
    frame_stride: int = 1,
    weather: str = "Clear",
    school_peak: bool = False,
    road_condition: str = "Normal",
    high_risk_zone: bool = False,
) -> tuple[str, str, Any, str]:
    """Annotate a video and return recommendation, trend data, and summary."""
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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = Path(output_file.name)

    writer = None
    frame_number = 0
    sample_count = 0
    last_car_count = 0
    records: list[dict] = []
    succeeded = False

    try:
        model = load_model()
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if frame_number % frame_stride == 0:
                result = model.predict(
                    source=frame,
                    classes=[PERSON_CLASS_ID, CAR_CLASS_ID, TRUCK_CLASS_ID],
                    verbose=False,
                )[0]
                boxes = result.boxes
                class_ids = boxes.cls.tolist() if boxes is not None else []
                people = sum(class_id == PERSON_CLASS_ID for class_id in class_ids)
                cars = sum(class_id == CAR_CLASS_ID for class_id in class_ids)
                trucks = sum(class_id == TRUCK_CLASS_ID for class_id in class_ids)

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

                last_car_count = cars
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
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create the processed video.")

            writer.write(annotated_frame)
            frame_number += 1

        if not frame_number or not records:
            raise ValueError("The uploaded video contains no readable frames.")

        timeline = pd.DataFrame(records)
        trend_plot = _make_trend_plot(timeline)
        succeeded = True

        speed = recommended_speed_kmh(last_car_count)
        speed_text = (
            f"## {speed} km/h\n"
            f"Demo recommendation · based on {last_car_count} cars in the last analyzed frame"
        )

        traffic_peak = timeline.loc[timeline["Cars"].idxmax()]
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
        summary = (
            "### Clip analysis\n"
            f"- Frames: {source_frames} · analyzed: {sample_count} · "
            f"interval: every {frame_stride} frame(s)\n"
            f"- Peak car detections: {int(traffic_peak['Cars'])} at "
            f"{float(traffic_peak['Time (seconds)']):.1f}s into the clip\n"
            f"- Lowest car detections: {int(traffic_lull['Cars'])} at "
            f"{float(traffic_lull['Time (seconds)']):.1f}s into the clip\n"
            f"- Peak people detections: {int(people_peak['People detected'])} "
            f"at {float(people_peak['Time (seconds)']):.1f}s into the clip\n"
            f"- Peak detected box area: "
            f"{float(timeline['Detected box area (%)'].max()):.1f}% of frame\n\n"
            "**Scenario inputs (simulated, not live feeds):** "
            f"{_context_summary(weather, school_peak, road_condition, high_risk_zone)}\n\n"
            f"**Scope:** {crowd_note} {timing_note}\n\n"
            "**Context is not used to change the demo speed threshold.** "
            "No live camera, weather service, school calendar, road-condition "
            "feed, accident data, or traffic sign is connected."
        )
        return str(output_path), speed_text, trend_plot, summary
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
            "trends. Weather, school, road, and zone inputs are scenario "
            "controls until approved local data feeds are available."
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
                            value=3,
                            step=1,
                            label="Analyze every Nth frame",
                            info="Higher values are faster; unsampled frames are not boxed.",
                        )
                        with gr.Accordion("Scenario controls (simulated)", open=True):
                            weather = gr.Radio(
                                ["Clear", "Rain", "Heavy rain"],
                                value="Clear",
                                label="Weather",
                            )
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
                        frame_stride,
                        weather,
                        school_peak,
                        road_condition,
                        high_risk_zone,
                    ],
                    outputs=[
                        output_video,
                        speed_output,
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
                    "- YOLOv8 detects cars, trucks, and people in sampled frames.\n"
                    "- The trend plot shows detections during this clip only. "
                    "It does not track unique people or infer crossing intent "
                    "or other pedestrian behavior.\n"
                    "- Detected box area is a rough image-occupancy proxy, "
                    "not calibrated road density.\n"
                    "- The 100/80/60 km/h thresholds are the original demo "
                    "rule, based only on car detections in the last analyzed "
                    "frame. Scenario controls do not alter it.\n\n"
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
