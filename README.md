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

Keep this terminal window open while using the app; the command runs the local
web server and is expected to keep running. Press `Ctrl+C` when you want to stop
the server.

Open the local URL printed by Gradio. Upload a video, enter the number of lanes
visible in the camera view, and choose how often to analyze frames. The output
video draws detections on analyzed frames; frames between samples remain
unannotated. It tracks vehicles across analyzed frames to estimate unique cars
and trucks in the clip; the estimate may miss or duplicate vehicles, especially
at larger frame intervals. The displayed demo speed uses the busiest analyzed
frame. Lower the frame interval for better tracking continuity. The first run
downloads `yolov8n.pt` from
[Ultralytics/YOLOv8 on the Hugging Face Hub](https://huggingface.co/Ultralytics/YOLOv8)
and the weather classifier from
[dima806/weather_types_image_detection](https://huggingface.co/dima806/weather_types_image_detection).
When automatic lane estimation is enabled, it also downloads
[NVIDIA SegFormer B0 fine-tuned on Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
to separate road from sidewalk before estimating lane lines. These models are
public; no token is needed.

The optional **Estimate visible lane count automatically** checkbox first uses
the Hugging Face Cityscapes segmenter to restrict line analysis to predicted
road pixels, reducing sidewalk/footpath edges. OpenCV estimates lane lines at
up to five points across the video. With two estimates, it uses the lower count
to avoid a high outlier; with more estimates it requires close agreement.
Otherwise it falls back to the manual lane-count input and reports why. This is
still experimental: the segmenter is not validated for every camera angle, and
line estimates can be wrong with faded markings, occlusion, or unusual road
layouts. Verify the estimate. The first analysis can take longer while models
download and load. After that, models are cached locally. The app shows progress
for model loading and frame analysis. The default sampling interval is every
15th frame; increase sampling detail by lowering that value.

The app samples up to three video frames for weather classification and displays
the predicted weather and confidence. Its card declares Apache-2.0 and its model
card reports weather-image classification results, but this model has not been
validated on traffic-camera footage. Rain changes the demo output only when it
wins the majority of sampled frame predictions with at least 60% average
confidence.

## Optional Hugging Face training job

`train.py` fine-tunes YOLOv8n on the public
[UA-DETRAC-derived vehicle dataset](https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection)
and publishes the resulting weights to a model repo. The dataset Hub card
declares CC-BY-4.0; the script includes attribution in the output model card.
The dataset is not Qatar-specific. Its card reports more images than are
currently present in the pinned Hub revision, has no validation folder, and
contains orphaned training labels. The script filters to matching image/label
pairs, makes a reproducible validation split from the training images, and
leaves the test split for final evaluation.

After confirming you accept the dataset terms and possible compute charges,
authenticate with `hf auth login` and submit:

```bash
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN train.py \
  --epochs 20 \
  --dataset-repo thundarstrom/traffic-vehicle-detection \
  --model-repo Tawuff/tawwuf-traffic-yolov8n
```

The job installs the missing OpenCV system libraries in its container when
needed. This requires the job container to run with package-install privileges.
Jobs are billed while starting/running. At the documented A10G-large rate of
$1.50/hour, the three-hour timeout caps compute at about $4.50; actual cost
depends on runtime. The job publishes `best.pt` and test metrics to the model
repo. To use it in the app after the job succeeds, run:

```bash
TRAFFIC_MODEL_REPO=Tawuff/tawwuf-traffic-yolov8n python app.py
```

The completed experimental model is published at
[Tawuff/tawwuf-traffic-yolov8n](https://huggingface.co/Tawuff/tawwuf-traffic-yolov8n).
On its held-out test split it scored mAP50 0.318 and mAP50-95 0.220, so do not
assume fine-tuning improved the pretrained detector; compare it on your own
authorized videos before choosing it.

The fine-tuned detector covers cars, buses, trucks, and vans, but not people.
When using it, pedestrian detections are unavailable. It is not validated for
Qatar roads or operational speed-control use.

The **Explore HF datasets** section searches Hugging Face's dataset catalog and
shows any license declared in each dataset card. It does not automatically
download datasets or treat search results as approved for Qatar.

For repeatable tests, select one of the sample videos in the **Analyze a
video** section and click **Load selected sample**. The first selection downloads
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
- YOLO's ByteTrack tracker estimates unique cars and trucks seen across analyzed
  frames. This is approximate, not a guaranteed count: skipped frames and
  occlusion can make the tracker miss or duplicate vehicles.
- Automatic lane estimation uses Cityscapes road/sidewalk segmentation followed
  by OpenCV line geometry, not a dedicated lane-marking model. Two frame
  estimates use the lower count; highly inconsistent estimates fall back to the
  manually entered lane count.
- The speed calculation divides the car detections in the busiest analyzed
  frame by the manually entered lane count or automatic estimate, then applies
  the demo thresholds per lane: fewer than 5 cars/lane = 100 km/h, 5–15 = 80
  km/h, more than 15 = 60 km/h. A confident majority rain estimate lowers this
  demo result one band (100→80, 80→60, 60 stays 60).
- Weather classification can be wrong due to scene, lighting, camera angle, or
  model-domain differences. The school, road-condition, and high-risk-zone
  controls are simulated scenarios, not live feeds.
- Dataset search may return data from other countries or with restrictive
  terms. Review provenance, license, privacy, and local relevance before
  research or use.
- This is not a traffic-management system. It has no government-camera
  connection, accident detector, or sign-control output. Operational use would
  require authorized data, local validation, privacy review, traffic-engineer
  approval, and fail-safe controls.

Use only video you are authorized to process.
