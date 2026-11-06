---
title: Tawwuf Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
---

# Tawwuf Traffic Intelligence

An English-language Hugging Face Spaces demo for exploring vehicle and
pedestrian-count trends in an uploaded traffic clip.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio. Upload a video and choose how often to
analyze frames. The output video draws detections on analyzed frames; frames
between samples remain unannotated. The first run downloads `yolov8n.pt` from
[Ultralytics/YOLOv8 on the Hugging Face Hub](https://huggingface.co/Ultralytics/YOLOv8).
This public model does not require a token.

The **Explore HF datasets** tab searches Hugging Face's dataset catalog and
shows any license declared in each dataset card. It does not automatically
download datasets or treat search results as approved for Qatar.

For repeatable tests, select one of the sample videos in the **Analyze a
video** tab and click **Load selected sample**. The first selection downloads
that MP4 from the public
[Small Crowd and Traffic Video Footages dataset](https://huggingface.co/datasets/parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages);
subsequent runs use the Hugging Face cache. Its Hub metadata declares MIT, but
the dataset card has little provenance detail, so verify its terms before use
beyond testing.

## Hugging Face Space

The Space metadata is at the top of this `README.md`. The public model and
public dataset search need no token. Never commit a Hugging Face token. If
future private/gated data requires authentication, configure it as a Space
secret and review the source dataset's access terms first.

## Demo limits and safety

- Traffic and people trends describe only the uploaded clip. Peak times are
  elapsed clip time, not daily commute or school-hour patterns.
- Person detections are approximate counts, not unique-person tracking or
  pedestrian-behavior understanding. Detected box area is not calibrated road
  density.
- Weather, school, road-condition, and high-risk-zone controls are manually
  selected scenarios, not live feeds. They do not change the original demo
  speed rule: fewer than 5 cars = 100 km/h, 5–15 = 80 km/h, more than 15 =
  60 km/h.
- Dataset search may return data from other countries or with restrictive
  terms. Review provenance, license, privacy, and local relevance before
  research or use.
- This is not a traffic-management system. It has no government-camera
  connection, accident detector, or sign-control output. Operational use would
  require authorized data, local validation, privacy review, traffic-engineer
  approval, and fail-safe controls.

Use only video you are authorized to process.
