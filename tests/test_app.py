import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
            (8, 2, False): 100,
            (10, 2, False): 80,
            (32, 2, False): 60,
            (4, 1, True): 80,
            (5, 1, True): 60,
            (20, 1, True): 60,
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

    def test_rain_is_applied_only_on_confident_majority(self):
        label, confidence, rain_detected = app._summarize_weather(
            [
                {"label": "rain", "score": 0.8},
                {"label": "cloudy", "score": 0.9},
                {"label": "rain", "score": 0.7},
            ]
        )
        self.assertEqual(label, "Rain")
        self.assertAlmostEqual(confidence, 0.75)
        self.assertTrue(rain_detected)

        _, _, rain_detected = app._summarize_weather(
            [
                {"label": "rain", "score": 0.55},
                {"label": "rain", "score": 0.5},
                {"label": "rain", "score": 0.6},
            ]
        )
        self.assertFalse(rain_detected)

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

    @patch("cv2.VideoWriter_fourcc", return_value=123)
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("app._make_trend_plot", return_value="plot")
    @patch("app._classify_weather_frame", return_value={"label": "cloudy", "score": 0.9})
    @patch("app.load_weather_classifier")
    @patch("app.load_model")
    def test_video_tracks_unique_vehicles_and_uses_peak_count_for_speed(
        self,
        load_model,
        _load_weather_classifier,
        _classify_weather_frame,
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
                (16, list(range(16))),
                (2, [0, 1]),
                (1, [0]),
            ]
        )

        class FakeBoxes:
            def __init__(self, count, ids):
                self.cls = SimpleNamespace(tolist=lambda: [app.CAR_CLASS_ID] * count)
                self.xyxy = SimpleNamespace(
                    tolist=lambda: [(0, 0, 1, 1)] * count
                )
                self.id = SimpleNamespace(
                    int=lambda: SimpleNamespace(tolist=lambda: ids)
                )

            def __len__(self):
                return len(self.cls.tolist())

        class FakeModel:
            def __init__(self):
                self.names = {0: "person", 2: "car", 7: "truck"}

            def track(self, **_kwargs):
                count, ids = next(detections)
                return [
                    SimpleNamespace(
                        boxes=FakeBoxes(count, ids),
                        plot=lambda: frames[0],
                    )
                ]

        load_model.return_value = FakeModel()

        output_path, speed_text, _weather, _plot, summary = app.process_video(
            "input.mp4",
            lane_count=1,
            frame_stride=1,
            progress=Mock(),
        )
        try:
            self.assertIn("60 km/h", speed_text)
            self.assertIn("Peak sampled frame: 16 cars", speed_text)
            self.assertIn("16 cars, 0 trucks/buses/vans", speed_text)
            self.assertIn("Counts are approximate", summary)
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
    @patch("app._classify_weather_frame", return_value={"label": "cloudy", "score": 0.9})
    @patch("app.load_weather_classifier")
    @patch("app.load_model")
    def test_automatic_lane_estimate_is_used_and_reported(
        self,
        load_model,
        _load_weather_classifier,
        _classify_weather_frame,
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
            SimpleNamespace(boxes=FakeBoxes(), plot=lambda: frame)
        ]

        output_path, speed_text, _weather, _plot, summary = app.process_video(
            "input.mp4",
            lane_count=1,
            frame_stride=1,
            detect_lanes_automatically=True,
            progress=Mock(),
        )
        try:
            self.assertIn("12 cars across 4 lane(s)", speed_text)
            self.assertIn(
                "Lane count: 4 (automatic road-area estimate from 2 frame(s))",
                speed_text,
            )
            self.assertIn("12 cars, 0 trucks/buses/vans", speed_text)
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
