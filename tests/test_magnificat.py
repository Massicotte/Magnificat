"""Tests de fumée pour Magnificat."""
import numpy as np


def test_overlays_render():
    import cv2

    from magnificat.overlays import (
        draw_dairy_bar,
        draw_halo,
        draw_hallelujah,
        draw_nudenet_panel,
        draw_sancta,
    )

    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.circle(frame, (640, 420), 150, (140, 110, 95), -1)
    display = frame.copy()
    cx, cy, r = draw_halo(display, frame, (540, 220, 740, 520))
    draw_sancta(display, cx, max(20, cy - r - 16), 1.2)
    draw_dairy_bar(display, 0.5)
    draw_nudenet_panel(display, [{"label": "FEMALE_NIPPLE", "score": 0.9}], lambda l: l)
    draw_hallelujah(display, 0.5)
    assert display.shape == frame.shape
    assert display.mean() != frame.mean()


def test_halo_occludes_behind_head():
    """L'arrière de l'auréole doit passer derrière le crâne : les pixels du crâne
    ne doivent pas être recouverts par l'or de l'anneau."""
    import cv2

    from magnificat.overlays import draw_halo

    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.circle(frame, (640, 420), 150, (140, 110, 95), -1)
    display = frame.copy()
    draw_halo(display, frame, (540, 320, 740, 620))
    skull_center = display[415, 640]
    original_center = frame[415, 640]
    assert all(abs(int(a) - int(b)) < 8 for a, b in zip(skull_center, original_center))


def test_hallelujah_sound_synth():
    from magnificat.sound import HallelujahSound

    s = HallelujahSound()
    s.play()
    assert s._thread is not None


def test_nudenet_labels_fr():
    from magnificat.analyzers import NudeNetAnalyzer

    assert NudeNetAnalyzer.label_fr("FEMALE_NIPPLE") == "téton"
    assert NudeNetAnalyzer.label_fr("BUTTOCKS_EXPOSED") == "fesses (exposées)"
    assert NudeNetAnalyzer.label_fr("INCONNU") == "INCONNU"


def test_yolo_bias_promotes_others():
    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class Vec:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class Box:
        def __init__(self, cls, c, xyxy):
            self.cls = Scalar(cls)
            self.conf = Scalar(c)
            self.xyxy = [Vec(xyxy)]

    class R:
        names = {0: "person", 1: "cow"}
        boxes = [Box(0, 0.3, [1, 2, 3, 4])]

    class FakeModel:
        names = {0: "person", 1: "cow"}

        def predict(self, bgr, conf=0.25, verbose=False):
            return [R()]

        def set_classes(self, classes):
            pass

    from magnificat.analyzers import YoloWorldAnalyzer

    analyzer = YoloWorldAnalyzer.__new__(YoloWorldAnalyzer)
    analyzer._model = FakeModel()
    analyzer._lock = None
    analyzer.conf_threshold = 0.25
    import numpy as np  # noqa: F401
    detections, others = analyzer.detect(np.zeros((10, 10, 3), np.uint8), bias=0.0)
    assert detections == [] and len(others) == 1
    detections, _ = analyzer.detect(np.zeros((10, 10, 3), np.uint8), bias=0.5)
    assert len(detections) == 1 and detections[0][0] == "lait_divin"
