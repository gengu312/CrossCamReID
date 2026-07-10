from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crosscam_mvp import VideoFileSource, resolve_video_paths, video_frame_interval  # noqa: E402


def write_test_video(path: Path, frame_count: int, fps: float = 20.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (320, 180),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create test video: {path}")
    try:
        for frame_index in range(frame_count):
            frame = np.full((180, 320, 3), (35, 38, 42), dtype=np.uint8)
            x = 20 + frame_index * 8
            cv2.rectangle(frame, (x, 70), (x + 90, 90), (40, 210, 235), -1)
            writer.write(frame)
    finally:
        writer.release()


class VideoReplayTest(unittest.TestCase):
    def test_video_paths_must_be_contiguous(self) -> None:
        args = SimpleNamespace(video_a="", video_b="b.avi", video_c="")
        with self.assertRaisesRegex(RuntimeError, "video-a"):
            resolve_video_paths(args)

        args = SimpleNamespace(video_a="a.avi", video_b="", video_c="c.avi")
        with self.assertRaisesRegex(RuntimeError, "video-b"):
            resolve_video_paths(args)

        args = SimpleNamespace(video_a="a.avi", video_b="b.avi", video_c="c.avi")
        self.assertEqual([path.name for path in resolve_video_paths(args)], ["a.avi", "b.avi", "c.avi"])

    def test_video_source_can_loop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crosscam-video-source-") as temp_dir:
            video_path = Path(temp_dir) / "loop.avi"
            write_test_video(video_path, frame_count=3, fps=20.0)
            source = VideoFileSource(video_path, loop=True)
            try:
                self.assertAlmostEqual(video_frame_interval([source]), 0.05, places=3)
                for _ in range(8):
                    ok, frame = source.read()
                    self.assertTrue(ok)
                    self.assertIsNotNone(frame)
            finally:
                source.release()

    def test_headless_dual_video_stops_at_shorter_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crosscam-video-replay-") as temp_dir:
            temp_root = Path(temp_dir)
            video_a = temp_root / "camera a.avi"
            video_b = temp_root / "camera b.avi"
            log_dir = temp_root / "logs"
            write_test_video(video_a, frame_count=12)
            write_test_video(video_b, frame_count=8)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "src" / "crosscam_mvp.py"),
                    "--video-a",
                    str(video_a),
                    "--video-b",
                    str(video_b),
                    "--headless",
                    "--log-dir",
                    str(log_dir),
                    "--warmup-frames",
                    "2",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("已打开离线视频", result.stdout)
            self.assertIn("离线视频播放结束", result.stdout)
            self.assertIn("已处理帧数：8", result.stdout)
            event_logs = list(log_dir.glob("*-events.csv"))
            self.assertEqual(len(event_logs), 1)
            with event_logs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            run_config = next(row for row in rows if row["event_type"] == "run_config")
            self.assertIn("source_mode=video", run_config["message"])
            self.assertIn("camera_count=2", run_config["message"])
            self.assertIn(f"video_a_path={video_a.resolve()}", run_config["message"])
            self.assertIn("video_a_fps=20.000", run_config["message"])
            self.assertIn("video_b_frames=8", run_config["message"])

            manifest_path = log_dir / "run_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["processed_frames"], 8)
            self.assertFalse(manifest["cross_camera_match_observed"])
            self.assertEqual(manifest["source"]["mode"], "video")
            self.assertEqual(len(manifest["source"]["videos"]), 2)
            self.assertEqual(manifest["source"]["videos"][0]["path"], str(video_a.resolve()))
            self.assertEqual(manifest["source"]["videos"][1]["frame_count"], 8)
            self.assertEqual(manifest["detector"]["type"], "motion")
            self.assertEqual(Path(manifest["event_log"]), event_logs[0].resolve())

    def test_headless_video_loop_honors_frame_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crosscam-video-loop-") as temp_dir:
            video_path = Path(temp_dir) / "loop.avi"
            write_test_video(video_path, frame_count=3)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "src" / "crosscam_mvp.py"),
                    "--video-a",
                    str(video_path),
                    "--loop-videos",
                    "--frames",
                    "10",
                    "--headless",
                    "--no-log",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("已处理帧数：10", result.stdout)


if __name__ == "__main__":
    unittest.main()
