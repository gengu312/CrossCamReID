from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.crosscam_mvp import (
    CrossCameraTracker,
    Detection,
    HybridDetector,
    TargetProfile,
    Track,
    apply_target_profile_with_hybrid_fallback,
    build_detectors,
    hybrid_detector_stats,
)


FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def make_detection(
    camera_id: int,
    feature: tuple[float, float],
    score: float = 0.9,
    bbox: tuple[int, int, int, int] = (4, 4, 12, 6),
) -> Detection:
    x, y, width, height = bbox
    return Detection(
        camera_id=camera_id,
        bbox=bbox,
        center=(x + width / 2.0, y + height / 2.0),
        area=float(width * height),
        score=score,
        feature=np.asarray(feature, dtype=np.float32),
        crop=np.zeros((height, width, 3), dtype=np.uint8),
    )


class FakeDetector:
    def __init__(self, camera_id: int, detections: list[Detection]) -> None:
        self.camera_id = camera_id
        self.detections = detections
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return list(self.detections)


def hybrid_args(**overrides):
    values = {
        "target_threshold": 0.8,
        "target_update_alpha": 0.0,
        "track_all_after_register": False,
        "target_stick_distance": 120.0,
        "target_switch_margin": 0.08,
        "target_sample_min_similarity": 0.72,
        "target_sample_min_interval": 0.8,
        "target_sample_max_count": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def registered_profile(camera_id: int = 0) -> tuple[TargetProfile, CrossCameraTracker]:
    profile = TargetProfile()
    profile.register_from_detection(make_detection(camera_id, (1.0, 0.0)))
    tracker = CrossCameraTracker(camera_count=1)
    tracker.activate_registered_target(1)
    tracker.pending_initial_target_camera = camera_id
    return profile, tracker


class HybridDetectorTests(unittest.TestCase):
    def test_build_hybrid_detectors_shares_models_between_cameras(self) -> None:
        args = SimpleNamespace(
            detector="hybrid",
            hybrid_fallback_interval=15,
            rfdetr_num_classes=1,
            rfdetr_conf=0.25,
            rfdetr_category_id_offset=1,
            yolo_classes="",
            rfdetr_classes="0",
        )

        def make_detector(_args, camera_id, _classes, model=None):
            return SimpleNamespace(
                camera_id=camera_id,
                model=object() if model is None else model,
                detect=lambda _frame: [],
            )

        with (
            patch("src.crosscam_mvp.create_yolo_detector", side_effect=make_detector),
            patch("src.crosscam_mvp.create_rfdetr_detector", side_effect=make_detector),
        ):
            detectors = build_detectors(args, camera_count=3)

        self.assertEqual(len(detectors), 3)
        self.assertIs(detectors[0].primary.model, detectors[1].primary.model)
        self.assertIs(detectors[1].primary.model, detectors[2].primary.model)
        self.assertIs(detectors[0].fallback.model, detectors[1].fallback.model)
        self.assertIs(detectors[1].fallback.model, detectors[2].fallback.model)

    def test_fallback_respects_interval(self) -> None:
        primary = FakeDetector(0, [])
        fallback = FakeDetector(0, [make_detection(0, (1.0, 0.0))])
        detector = HybridDetector(primary, fallback, fallback_interval=3)

        detector.detect(FRAME)
        self.assertEqual(len(detector.detect_fallback(FRAME)), 1)
        detector.detect(FRAME)
        self.assertEqual(detector.detect_fallback(FRAME), [])
        detector.detect(FRAME)
        self.assertEqual(detector.detect_fallback(FRAME), [])
        detector.detect(FRAME)
        self.assertEqual(len(detector.detect_fallback(FRAME)), 1)
        self.assertEqual(detector.fallback_calls, 2)

    def test_inactive_target_never_uses_fallback(self) -> None:
        primary = FakeDetector(0, [])
        fallback = FakeDetector(0, [make_detection(0, (1.0, 0.0))])
        detector = HybridDetector(primary, fallback, fallback_interval=1)
        raw = [detector.detect(FRAME)]

        filtered = apply_target_profile_with_hybrid_fallback(
            raw,
            [FRAME],
            [detector],
            TargetProfile(),
            CrossCameraTracker(camera_count=1),
            hybrid_args(),
            Path("runs_test"),
            1.0,
        )

        self.assertEqual(filtered, [[]])
        self.assertEqual(detector.fallback_calls, 0)

    def test_primary_target_match_skips_fallback(self) -> None:
        profile, tracker = registered_profile()
        primary = FakeDetector(0, [make_detection(0, (1.0, 0.0))])
        fallback = FakeDetector(0, [make_detection(0, (1.0, 0.0))])
        detector = HybridDetector(primary, fallback, fallback_interval=1)
        raw = [detector.detect(FRAME)]

        filtered = apply_target_profile_with_hybrid_fallback(
            raw,
            [FRAME],
            [detector],
            profile,
            tracker,
            hybrid_args(),
            Path("runs_test"),
            1.0,
        )

        self.assertTrue(filtered[0][0].is_target_match)
        self.assertEqual(detector.fallback_calls, 0)

    def test_fallback_recovers_only_matching_target(self) -> None:
        profile, tracker = registered_profile()
        primary = FakeDetector(0, [make_detection(0, (0.0, 1.0))])
        fallback = FakeDetector(
            0,
            [
                make_detection(0, (0.0, 1.0), score=0.95),
                make_detection(0, (1.0, 0.0), score=0.70),
            ],
        )
        detector = HybridDetector(primary, fallback, fallback_interval=1)
        raw = [detector.detect(FRAME)]

        filtered = apply_target_profile_with_hybrid_fallback(
            raw,
            [FRAME],
            [detector],
            profile,
            tracker,
            hybrid_args(),
            Path("runs_test"),
            1.0,
        )

        self.assertEqual(len(filtered[0]), 1)
        self.assertTrue(filtered[0][0].is_target_match)
        self.assertAlmostEqual(filtered[0][0].score, 0.70)
        self.assertEqual(detector.fallback_calls, 1)
        self.assertEqual(detector.fallback_candidates, 2)
        self.assertEqual(detector.fallback_accepted, 1)
        self.assertEqual(
            hybrid_detector_stats([detector]),
            {"calls": 1, "candidates": 2, "accepted": 1},
        )

    def test_same_camera_fallback_rejects_large_position_jump(self) -> None:
        profile, tracker = registered_profile()
        registered = make_detection(0, (1.0, 0.0))
        tracker.active[0].append(
            Track(
                camera_id=0,
                local_id=1,
                global_id=1,
                bbox=registered.bbox,
                center=registered.center,
                feature=registered.feature,
                last_seen=1.0,
            )
        )
        primary = FakeDetector(0, [make_detection(0, (0.0, 1.0))])
        fallback = FakeDetector(0, [make_detection(0, (1.0, 0.0), bbox=(300, 300, 12, 6))])
        detector = HybridDetector(primary, fallback, fallback_interval=1)

        filtered = apply_target_profile_with_hybrid_fallback(
            [detector.detect(FRAME)],
            [FRAME],
            [detector],
            profile,
            tracker,
            hybrid_args(target_stick_distance=80.0),
            Path("runs_test"),
            1.0,
        )

        self.assertEqual(filtered, [[]])
        self.assertEqual(detector.fallback_calls, 1)
        self.assertEqual(detector.fallback_accepted, 0)


if __name__ == "__main__":
    unittest.main()
