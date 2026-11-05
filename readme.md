# Tawoon

## Count cars and trucks in a video

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The script downloads `yolov8n.pt` from the public
[Ultralytics/YOLOv8 repository on Hugging Face Hub](https://huggingface.co/Ultralytics/YOLOv8)
on the first run and caches it locally. No Hugging Face token is needed for this
public model.

Use a short MP4 clip that you have permission to use. Create a `data` folder,
put the video there, then run:

```bash
python app.py data/sample-video.mp4
```

The script prints CSV rows to stdout, with one row per video frame:

```text
frame,car_count,truck_count,total_count
1,3,1,4
2,2,1,3
```

You can save the results to a CSV file with shell redirection:

```bash
python app.py data/sample-video.mp4 > counts.csv
```
