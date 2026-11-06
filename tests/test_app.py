import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import app
import train


class AppTests(unittest.TestCase):
    def test_recommended_speed_boundaries(self):
        cases = {
            0: 100,
            4: 100,
            5: 80,
            15: 80,
            16: 60,
            (8, 2): 100,
            (10, 2): 80,
            (32, 2): 60,
            (4, 1, "Roadworks"): 80,
            (5, 1, "Wet road"): 60,
            (4, 1, "Reduced visibility"): 80,
            (20, 1, "Roadworks"): 60,
        }
        for inputs, expected in cases.items():
            arguments = inputs if isinstance(inputs, tuple) else (inputs,)
            with self.subTest(arguments=arguments):
                self.assertEqual(app.recommended_speed_kmh(*arguments), expected)

    def test_lane_count_must_be_a_positive_integer(self):
        for lane_count in (0, -1, 1.5, None):
            with self.subTest(lane_count=lane_count), self.assertRaisesRegex(
                ValueError, "Lane count"
            ):
                app.recommended_speed_kmh(10, lane_count)

    def test_training_split_is_reproducible_and_keeps_both_sets(self):
        image_paths = [Path(f"image-{index}.jpg") for index in range(10)]
        training, validation = train.split_training_images(image_paths, seed=7)
        repeat_training, repeat_validation = train.split_training_images(
            image_paths, seed=7
        )
        self.assertEqual(training, repeat_training)
        self.assertEqual(validation, repeat_validation)
        self.assertEqual(len(training), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(set(training) & set(validation))

    def test_training_data_prep_filters_orphan_labels_and_makes_test_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            for split in ("train", "test"):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir(parents=True)
            for index in range(5):
                (source / "train" / "images" / f"train-{index}.jpg").write_bytes(
                    b"image"
                )
                (source / "train" / "labels" / f"train-{index}.txt").write_text(
                    "0 0.5 0.5 0.25 0.25\n",
                    encoding="utf-8",
                )
            (source / "train" / "labels" / "orphan.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n",
                encoding="utf-8",
            )
            (source / "test" / "images" / "test.jpg").write_bytes(b"image")
            (source / "test" / "labels" / "test.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n",
                encoding="utf-8",
            )

            output = root / "prepared"
            counts = train.prepare_dataset(source, output, seed=7)

            self.assertEqual(counts["train_images"], 4)
            self.assertEqual(counts["validation_images"], 1)
            self.assertEqual(counts["test_images"], 1)
            self.assertEqual(counts["orphan_train_labels"], 1)
            self.assertTrue((output / "dataset.yaml").is_file())
            self.assertEqual(len(list((output / "train" / "images").iterdir())), 4)

    def test_estimates_six_lanes_from_two_synthetic_three_lane_roadways(self):
        import cv2
        import numpy as np

        frame = np.zeros((360, 1280, 3), dtype=np.uint8)
        for bottom_x in (100, 220, 340, 460, 820, 940, 1060, 1180):
            cv2.line(frame, (640, 175), (bottom_x, 340), (255, 255, 255), 3)

        self.assertEqual(app._estimate_visible_lane_count(frame), 6)

    def test_lane_estimation_returns_none_without_road_markings(self):
        import numpy as np

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertIsNone(app._estimate_visible_lane_count(frame))

    def test_road_mask_restricts_lane_line_search(self):
        import cv2
        import numpy as np

        frame = np.zeros((360, 1280, 3), dtype=np.uint8)
        for bottom_x in (100, 220, 340, 460, 820, 940, 1060, 1180):
            cv2.line(frame, (640, 175), (bottom_x, 340), (255, 255, 255), 3)

        road_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.assertIsNone(
            app._estimate_visible_lane_count(frame, road_mask=road_mask)
        )

    def test_road_mask_dimensions_must_match_frame(self):
        import numpy as np

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "match the video frame"):
            app._estimate_visible_lane_count(
                frame,
                road_mask=np.zeros((100, 100), dtype=np.uint8),
            )

    def test_lane_estimate_requires_consistent_frames_and_breaks_ties_low(self):
        self.assertEqual(app._select_consistent_lane_estimate([4, 6]), 4)
        self.assertEqual(app._select_consistent_lane_estimate([4, 5, 5]), 5)
        self.assertIsNone(app._select_consistent_lane_estimate([4, 6, 7]))
        self.assertIsNone(app._select_consistent_lane_estimate([4]))

    def test_speed_rejects_unlisted_road_conditions(self):
        with self.assertRaisesRegex(ValueError, "listed road conditions"):
            app.recommended_speed_kmh(4, road_condition="Storm")

    def test_weather_factors_reduce_speed_and_duplicate_conditions_count_once(self):
        self.assertEqual(
            app.recommended_speed_kmh(
                4,
                lane_count=4,
                weather_factors=("precipitation",),
            ),
            80,
        )
        self.assertEqual(
            app.recommended_speed_kmh(
                4,
                lane_count=4,
                road_condition="Wet road",
                weather_factors=("precipitation",),
            ),
            80,
        )
        self.assertEqual(
            app.recommended_speed_kmh(
                4,
                lane_count=4,
                road_condition="Roadworks",
                weather_factors=("precipitation",),
            ),
            60,
        )

    @patch("app.urlopen")
    def test_open_meteo_geocodes_location_and_returns_current_weather(self, open_url):
        open_url.side_effect = [
            BytesIO(
                json.dumps(
                    {
                        "results": [
                            {
                                "name": "Doha",
                                "admin1": "Doha",
                                "country": "Qatar",
                                "latitude": 25.29,
                                "longitude": 51.53,
                                "timezone": "Asia/Qatar",
                            }
                        ]
                    }
                ).encode()
            ),
            BytesIO(
                json.dumps(
                    {
                        "timezone": "Asia/Qatar",
                        "current": {
                            "temperature_2m": 35.2,
                            "precipitation": 1.4,
                            "rain": 1.4,
                            "snowfall": 0.0,
                            "weather_code": 61,
                            "wind_speed_10m": 12.0,
                        },
                    }
                ).encode()
            ),
        ]

        weather = app.fetch_open_meteo_weather("Doha, Qatar")

        self.assertEqual(weather["location"], "Doha, Qatar")
        self.assertEqual(weather["condition"], "Rain")
        self.assertEqual(weather["temperature_c"], 35.2)
        self.assertEqual(weather["adjustment_factors"], {"precipitation"})
        self.assertEqual(open_url.call_count, 2)
        self.assertIn(
            "geocoding-api.open-meteo.com",
            urlparse(open_url.call_args_list[0].args[0]).netloc,
        )
        forecast_query = parse_qs(urlparse(open_url.call_args_list[1].args[0]).query)
        self.assertEqual(forecast_query["wind_speed_unit"], ["kmh"])
        self.assertIn("weather_code", forecast_query["current"][0])

    @patch("app.urlopen")
    def test_open_meteo_reports_unknown_location(self, open_url):
        open_url.return_value = BytesIO(json.dumps({"results": None}).encode())

        with self.assertRaisesRegex(ValueError, "could not find a location"):
            app.fetch_open_meteo_weather("Atlantis")

    def test_context_is_explicitly_a_scenario(self):
        context = app._context_summary(True, "Roadworks", False)
        self.assertIn("School-peak scenario: On", context)
        self.assertIn("Road scenario: Roadworks", context)

    def test_video_path_accepts_gradio_video_object(self):
        self.assertEqual(
            app._resolve_video_path({"video": "/tmp/sample.mp4", "subtitles": None}),
            "/tmp/sample.mp4",
        )
        self.assertEqual(
            app._resolve_video_path(
                {"video": {"path": "/tmp/sample.mp4"}, "subtitles": None}
            ),
            "/tmp/sample.mp4",
        )

    def test_video_path_requires_a_video(self):
        with self.assertRaisesRegex(ValueError, "Upload or load"):
            app._resolve_video_path(None)

    def test_video_rejects_unknown_road_condition(self):
        with self.assertRaisesRegex(ValueError, "listed road conditions"):
            app.process_video("input.mp4", road_condition="Storm")

    def test_video_rejects_non_integer_frame_interval(self):
        for frame_stride in (0, -1, 1.5):
            with self.subTest(frame_stride=frame_stride), self.assertRaisesRegex(
                ValueError, "positive whole number"
            ):
                app.process_video("input.mp4", frame_stride=frame_stride)

    @patch("cv2.VideoWriter_fourcc", return_value=123)
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("app._make_trend_plot", return_value="plot")
    @patch(
        "app.fetch_open_meteo_weather",
        return_value={
            "location": "Doha, Qatar",
            "timezone": "Asia/Qatar",
            "condition": "Rain",
            "weather_code": 61,
            "temperature_c": 30.0,
            "precipitation_mm": 1.0,
            "rain_mm": 1.0,
            "snowfall_cm": 0.0,
            "wind_speed_kmh": 20.0,
            "adjustment_factors": {"precipitation"},
        },
    )
    @patch("app.load_model")
    def test_video_counts_all_vehicle_classes_and_uses_peak_count_for_speed(
        self,
        load_model,
        fetch_weather,
        _make_trend_plot,
        video_capture,
        video_writer,
        video_writer_fourcc,
    ):
        import cv2
        import numpy as np

        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30 if prop == cv2.CAP_PROP_FPS else len(frames)
        )
        capture.read.side_effect = [(True, frame) for frame in frames] + [
            (False, None)
        ]
        video_capture.return_value = capture

        writer = Mock()
        writer.isOpened.return_value = True
        video_writer.return_value = writer

        detections = iter(
            [
                ([0] * 12 + [1, 2, 3, 1, 4], list(range(17))),
                ([1, 1], [0, 12]),
                ([1], [0]),
            ]
        )

        class FakeBoxes:
            def __init__(self, class_ids, ids):
                self.cls = SimpleNamespace(tolist=lambda: class_ids)
                self.xyxy = SimpleNamespace(
                    tolist=lambda: [(0, 0, 1, 1)] * len(class_ids)
                )
                self.id = SimpleNamespace(
                    int=lambda: SimpleNamespace(tolist=lambda: ids)
                )

            def __len__(self):
                return len(self.cls.tolist())

        class FakeModel:
            def __init__(self):
                self.names = {
                    0: "car",
                    1: "bus",
                    2: "truck",
                    3: "van",
                    4: "person",
                }
                self.track_call_count = 0

            def track(self, **_kwargs):
                self.uses_botsort = _kwargs["tracker"] == "botsort.yaml"
                self.track_call_count += 1
                class_ids, ids = next(detections)
                return [
                    SimpleNamespace(
                        boxes=FakeBoxes(class_ids, ids),
                        names=self.names,
                        plot=lambda: frames[0],
                    )
                ]

        load_model.return_value = FakeModel()

        output_path, speed_text, condition_text, _plot, summary = app.process_video(
            "input.mp4",
            lane_count=4,
            frame_stride=2,
            road_condition="Roadworks",
            weather_location="Doha, Qatar",
            progress=Mock(),
        )
        try:
            fetch_weather.assert_called_once_with("Doha, Qatar")
            load_model.assert_called_once_with()
            self.assertEqual(load_model.return_value.track_call_count, 2)
            self.assertTrue(load_model.return_value.uses_botsort)
            self.assertEqual(writer.write.call_count, len(frames))
            self.assertTrue(writer.write.call_args_list[1].args[0].any())
            self.assertIn("60 km/h", speed_text)
            self.assertIn("Detector: pretrained YOLOv8n (recommended)", speed_text)
            self.assertIn("Total unique vehicles in clip: 16", speed_text)
            self.assertIn("Peak sampled frame: 16 vehicles", speed_text)
            self.assertIn("12 cars, 4 buses/trucks/vans", speed_text)
            self.assertIn("Total unique vehicles in clip: 16", summary)
            self.assertIn("Tracked class totals: 12 cars, 4 buses/trucks/vans", summary)
            self.assertIn("analyzed: 2", summary)
            self.assertIn("detected every 2 frame(s)", summary)
            self.assertIn("Boxes are held between analyzed frames", summary)
            self.assertIn("Road condition scenario:** Roadworks", condition_text)
            self.assertIn("Current Open-Meteo weather for Doha, Qatar:** Rain", condition_text)
            self.assertIn("Weather speed factors applied: precipitation", condition_text)
            self.assertIn("Each distinct factor lowers", condition_text)
            self.assertIn("Detection model: pretrained YOLOv8n", summary)
            self.assertIn("Road scenario: Roadworks", summary)
            self.assertIn("estimates rather than ground truth", summary)
            video_writer_fourcc.assert_called_once_with(*"avc1")
            self.assertTrue(output_path.endswith(".mp4"))
        finally:
            app.Path(output_path).unlink(missing_ok=True)

    @patch("app._segment_road_surface")
    @patch("app.load_lane_scene_segmenter")
    @patch("app._estimate_visible_lane_count", side_effect=[4, 6])
    @patch("cv2.VideoWriter_fourcc", return_value=123)
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("app._make_trend_plot", return_value="plot")
    @patch("app.load_model")
    def test_automatic_lane_estimate_is_used_and_reported(
        self,
        load_model,
        _make_trend_plot,
        video_capture,
        video_writer,
        _video_writer_fourcc,
        estimate_lane_count,
        _load_lane_scene_segmenter,
        segment_road_surface,
    ):
        import cv2
        import numpy as np

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30 if prop == cv2.CAP_PROP_FPS else 2
        )
        capture.read.side_effect = [
            (True, frame),
            (True, frame),
            (False, None),
        ]
        video_capture.return_value = capture

        writer = Mock()
        writer.isOpened.return_value = True
        video_writer.return_value = writer

        class FakeBoxes:
            cls = SimpleNamespace(tolist=lambda: [0] * 12)
            xyxy = SimpleNamespace(tolist=lambda: [(0, 0, 1, 1)] * 12)
            id = SimpleNamespace(
                int=lambda: SimpleNamespace(tolist=lambda: list(range(12)))
            )

            def __len__(self):
                return 12

        load_model.return_value.names = {
            0: "car",
            1: "bus",
            2: "truck",
            3: "van",
        }
        load_model.return_value.track.return_value = [
            SimpleNamespace(
                boxes=FakeBoxes(),
                names=load_model.return_value.names,
                plot=lambda: frame,
            )
        ]

        output_path, speed_text, _weather, _plot, summary = app.process_video(
            "input.mp4",
            lane_count=1,
            frame_stride=1,
            detect_lanes_automatically=True,
            progress=Mock(),
        )
        try:
            self.assertIn("12 vehicles across 4 lane(s)", speed_text)
            self.assertIn(
                "Lane count: 4 (automatic road-area estimate from 2 frame(s))",
                speed_text,
            )
            self.assertIn("12 cars, 0 buses/trucks/vans", speed_text)
            self.assertIn("estimates: [4, 6]", summary)
            self.assertEqual(estimate_lane_count.call_count, 2)
            self.assertEqual(segment_road_surface.call_count, 2)
        finally:
            app.Path(output_path).unlink(missing_ok=True)

    def test_class_mapping_supports_pretrained_and_custom_yolo_heads(self):
        self.assertEqual(
            app._detection_class_ids(
                SimpleNamespace(names={0: "person", 2: "car", 5: "bus", 7: "truck"})
            ),
            ([0, 2, 5, 7], {0}, {2}, {5, 7}),
        )
        self.assertEqual(
            app._detection_class_ids(
                SimpleNamespace(names={0: "car", 1: "bus", 2: "truck", 3: "van"})
            ),
            ([0, 1, 2, 3], set(), {0}, {1, 2, 3}),
        )

    @patch("huggingface_hub.list_datasets")
    def test_dataset_search_returns_license_and_link(self, list_datasets):
        list_datasets.return_value = [
            SimpleNamespace(
                id="example/traffic",
                cardData={"license": "cc-by-4.0"},
                downloads=12,
            )
        ]

        results = app.search_hf_datasets("traffic video")

        self.assertEqual(results.iloc[0]["License in Hub card"], "cc-by-4.0")
        self.assertEqual(
            results.iloc[0]["Hugging Face page"],
            "https://huggingface.co/datasets/example/traffic",
        )
        list_datasets.assert_called_once()

    def test_dataset_search_requires_a_query(self):
        with self.assertRaisesRegex(ValueError, "Enter a topic"):
            app.search_hf_datasets("  ")

    @patch("huggingface_hub.hf_hub_download")
    def test_sample_download_uses_allowlisted_hub_file(self, download):
        download.return_value = "/tmp/crowd03.mp4"

        result = app.load_sample_video("Crowd sample 03 (2.5 MB)")

        self.assertEqual(result, "/tmp/crowd03.mp4")
        download.assert_called_once_with(
            repo_id=app.SAMPLE_VIDEO_REPO,
            repo_type="dataset",
            filename="crowd03.mp4",
        )

    def test_sample_download_rejects_unknown_clip(self):
        with self.assertRaisesRegex(ValueError, "listed sample"):
            app.load_sample_video("unknown clip")


if __name__ == "__main__":
    unittest.main()
