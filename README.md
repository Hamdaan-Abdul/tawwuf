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
visible in the camera view, and select the road-condition scenario. The app
uses pretrained YOLOv8n from the Hugging Face Hub; the separately fine-tuned
model is not used because it misclassifies cars as buses, trucks, or vans. The
detector and tracker run on every frame, and the output video keeps detections
visible throughout playback. The app reports an estimated unique-vehicle total
and the peak number present in any one frame. Full-frame tracking is slower
than sparse sampling, but avoids tracker gaps and makes the peak consistent.
The first run downloads `yolov8n.pt` from
[Ultralytics/YOLOv8 on the Hugging Face Hub](https://huggingface.co/Ultralytics/YOLOv8)
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
for model loading and frame analysis. Longer or higher-resolution videos take
longer because every frame is analyzed.

Automatic weather classification is disabled because it incorrectly labeled
clear traffic footage as rainy. The **Road condition** control is a manual
scenario, not an AI prediction. Selecting **Wet road**, **Roadworks**, or
**Reduced visibility** lowers the density-based demo recommendation by one
band (20 km/h, with a 60 km/h floor). **Normal** leaves the density-based
recommendation unchanged.

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
depends on runtime. The job publishes `best.pt` and test metrics to the model repo.

The completed experimental model is published at
[Tawuff/tawwuf-traffic-yolov8n](https://huggingface.co/Tawuff/tawwuf-traffic-yolov8n).
On its held-out test split it scored mAP50 0.318 and mAP50-95 0.220, so do not
assume fine-tuning improved the pretrained detector. Testing on the provided
traffic sample also showed it labeling many cars as buses; the underlying
dataset has a severe class imbalance. For that reason the app uses the
pretrained detector instead. Treat the fine-tuned model's labels and counts as
experimental; improving it requires representative, correctly labeled traffic
data and retraining.

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
- YOLO's BoT-SORT tracker estimates unique cars and other vehicles by tracking
  every frame. The peak and speed demo use all detected car, bus, truck, and
  van classes, and show their class breakdown. A track's displayed class is
  its most frequent label across frames to reduce label flicker. The unique
  total is still an estimate, not ground truth: occlusion or a lost/reassigned
  track ID can cause missed or duplicated vehicles. A model can also
  misclassify vehicles (as the fine-tuned model does on some sample footage).
- Automatic lane estimation uses Cityscapes road/sidewalk segmentation followed
  by OpenCV line geometry, not a dedicated lane-marking model. Two frame
  estimates use the lower count; highly inconsistent estimates fall back to the
  manually entered lane count.
- The speed calculation divides vehicle detections in the busiest frame by
  the manually entered lane count or automatic estimate, then applies
  the demo thresholds per lane: fewer than 5 vehicles/lane = 100 km/h, 5–15
  = 80 km/h, more than 15 = 60 km/h. Manually selecting wet road, roadworks,
  or reduced visibility lowers the demo result one band (100→80, 80→60,
  60 stays 60). Automatic weather classification is disabled because it was
  unreliable on traffic-camera footage. School, road-condition, and
  high-risk-zone controls are simulated scenarios, not live feeds.
- Dataset search may return data from other countries or with restrictive
  terms. Review provenance, license, privacy, and local relevance before
  research or use.
- This is not a traffic-management system. It has no government-camera
  connection, accident detector, or sign-control output. Operational use would
  require authorized data, local validation, privacy review, traffic-engineer
  approval, and fail-safe controls.

Use only video you are authorized to process.
