import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class AppTests(unittest.TestCase):
    def test_recommended_speed_boundaries(self):
        cases = {
            0: 100,
            4: 100,
            5: 80,
            15: 80,
            16: 60,
        }
        for car_count, expected in cases.items():
            with self.subTest(car_count=car_count):
                self.assertEqual(app.recommended_speed_kmh(car_count), expected)

    def test_context_is_explicitly_a_scenario(self):
        context = app._context_summary("Rain", True, "Roadworks", False)
        self.assertIn("Weather: Rain", context)
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
