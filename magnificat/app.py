#!/usr/bin/env python3
"""Magnificat — retour webcam sanctifié pour spectacle technocritique.

Fenêtre principale : vidéo plein écran sans déformation (bandes noires sur
les côtés), surimpressions NudeNet, YOLO-World (Lait divin), MediaPipe
(hallelujah) et auréoles + SANCTA au-dessus des têtes.
Fenêtre secondaire : slider « Foi » pour biaiser YOLO-World.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .analyzers import FaceTracker, HandSignalAnalyzer, NudeNetAnalyzer, YoloWorldAnalyzer
from .overlays import (
    draw_bias_indicators,
    draw_dairy_bar,
    draw_halo,
    draw_hallelujah,
    draw_nudenet_panel,
    draw_sancta,
    draw_two_fingers_flash,
)
from .sound import HallelujahSound

GOLD = (72, 215, 255)
NUDE_INTERVAL_S = 2.0
YOLO_INTERVAL_S = 1.0
HALLELUJAH_DISPLAY_S = 2.0
HALLELUJAH_COOLDOWN_S = 4.0
DAIRY_MAX = 12.0
HEAD_LABELS = {"FACE_MALE", "FACE_FEMALE"}
HEAD_MARGIN = 1.25
HALO_LIFETIME_S = 3.5
HALO_SMOOTHING = 0.35


class HaloTracker:
    """Suit les têtes (visages NudeNet) pour que les auréoles collent aux comédiennes."""

    def __init__(self):
        self.tracks = []

    def update(self, head_boxes, now):
        matched = []
        for hb in head_boxes:
            hx1, hy1, hx2, hy2 = hb
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            best, best_d = None, 1e9
            for tr in self.tracks:
                d = ((tr["cx"] - cx) ** 2 + (tr["cy"] - cy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d < 120:
                best["cx"] += (cx - best["cx"]) * HALO_SMOOTHING * 2
                best["cy"] += (cy - best["cy"]) * HALO_SMOOTHING * 2
                best["w"] = (hx2 - hx1) * HEAD_MARGIN
                best["h"] = (hy2 - hy1) * HEAD_MARGIN
                best["last_seen"] = now
                matched.append(best)
            else:
                track = {"cx": cx, "cy": cy, "w": (hx2 - hx1) * HEAD_MARGIN,
                         "h": (hy2 - hy1) * HEAD_MARGIN, "last_seen": now}
                self.tracks.append(track)
                matched.append(track)
        self.tracks = [t for t in self.tracks if now - t["last_seen"] < HALO_LIFETIME_S]
        return matched


class BiasWindow(QWidget):
    """Fenêtre secondaire : slider pour biaiser YOLO-World (« il en voit partout »)."""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setWindowTitle("Magnificat — Foi (biais YOLO-World)")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Foi du spectacle : plus il y en a, plus le modèle en voit partout"))
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_change)
        self.value_label = QLabel("0%")
        self.value_label.setMinimumWidth(48)
        row.addWidget(self.slider)
        row.addWidget(self.value_label)
        layout.addLayout(row)
        hint = QLabel("0% = comportement normal du modèle\n100% = toutes les détections deviennent du Lait divin")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        self.resize(420, 180)

    def on_change(self, value):
        self.main.set_bias(value / 100.0)
        self.value_label.setText(f"{value}%")


class VideoWidget(QWidget):
    """Retour vidéo plein écran, sans déformation (bandes noires sur les côtés)."""

    def __init__(self):
        super().__init__()
        self.frame_bgr = None

    def set_frame(self, frame_bgr):
        self.frame_bgr = frame_bgr

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.frame_bgr is None:
            return
        fh, fw = self.frame_bgr.shape[:2]
        cw, ch = self.width(), self.height()
        scale = min(cw / fw, ch / fh)
        dw, dh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        x, y = (cw - dw) // 2, (ch - dh) // 2
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, fw, fh, 3 * fw, QImage.Format_RGB888)
        painter.drawImage(
            x, y, image.scaled(dw, dh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        )


class MainWindow(QMainWindow):
    def __init__(self, camera_index=0, use_test_pattern=False):
        super().__init__()
        self.setWindowTitle("Magnificat — Sancta Lactatio")
        self.video = VideoWidget()
        self.setCentralWidget(self.video)

        self.sound = HallelujahSound()
        self.bias = 0.0
        self.hallelujah_started_at = None
        self.last_hallelujah_at = -1e9
        self.dairy_level = 0.0
        self.last_nude_time = -1e9
        self.last_yolo_time = -1e9
        self.nude_detections = []
        self.dairy_detections = []
        self.halo_tracker = HaloTracker()

        self.nude = self._try_init(NudeNetAnalyzer, "NudeNet")
        self.yolo = self._try_init(YoloWorldAnalyzer, "YOLO-World")
        self.hands = self._try_init(HandSignalAnalyzer, "MediaPipe Hands")
        self.faces = self._try_init(FaceTracker, "MediaPipe FaceMesh")

        self.use_test_pattern = use_test_pattern
        self.cap = None
        if not use_test_pattern:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la webcam {camera_index}")

        self.bias_window = BiasWindow(self)
        self.bias_window.show()

        self.showFullScreen()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    @staticmethod
    def _try_init(factory, name):
        try:
            return factory()
        except Exception as exc:
            print(f"[Magnificat] {name} indisponible : {exc}")
            return None

    def set_bias(self, value):
        self.bias = value

    def grab_frame(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        t = time.time()
        frame = np.zeros((720, 1280, 3), np.uint8)
        cv2.putText(frame, "Magnificat — pattern de test", (30, 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, GOLD, 2)
        x = int(640 + 300 * np.sin(t))
        cv2.circle(frame, (x, 360), 90, (120, 90, 80), -1)
        cv2.circle(frame, (x - 30, 340), 12, (200, 170, 160), -1)
        cv2.circle(frame, (x + 30, 340), 12, (200, 170, 160), -1)
        return frame

    def tick(self):
        frame = self.grab_frame()
        if frame is None:
            return
        now = time.time()

        two_fingers = False
        hand_points = []
        if self.hands is not None and self.hands.available:
            _, two, hand_points = self.hands.detect_two_fingers(frame)
            two_fingers = two
        if two_fingers and now - self.last_hallelujah_at > HALLELUJAH_COOLDOWN_S:
            self.last_hallelujah_at = now
            self.hallelujah_started_at = now
            self.sound.play()

        if now - self.last_yolo_time > YOLO_INTERVAL_S:
            self.last_yolo_time = now
            if self.yolo is not None:
                try:
                    detections, _others = self.yolo.detect(frame, self.bias)
                    self.dairy_detections = detections
                except Exception as exc:
                    print(f"[Magnificat] YOLO-World erreur : {exc}")
                    self.dairy_detections = []

        if now - self.last_nude_time > NUDE_INTERVAL_S:
            self.last_nude_time = now
            if self.nude is not None:
                try:
                    self.nude_detections = self.nude.detect(frame)
                except Exception as exc:
                    print(f"[Magnificat] NudeNet erreur : {exc}")
                    self.nude_detections = []

        display = frame.copy()
        draw_bias_indicators(display, self.bias, self.dairy_detections)
        for det in self.nude_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            label = det.get("label", "")
            color = (0, 255, 0) if label in HEAD_LABELS else (0, 200, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)

        head_boxes = [
            [det["box"][0], det["box"][1], det["box"][2], det["box"][3]]
            for det in self.nude_detections
            if det.get("label") in HEAD_LABELS
        ]
        if not head_boxes and self.faces is not None and self.faces.available:
            head_boxes = self.faces.head_boxes(frame)
        tracks = self.halo_tracker.update(head_boxes, now)
        for tr in tracks:
            w, h = tr["w"], tr["h"]
            cx, cy = int(tr["cx"]), int(tr["cy"])
            head_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            hcx, hcy, r = draw_halo(display, frame, head_box)
            draw_sancta(display, hcx, max(20, hcy - r - 16), scale=max(0.7, r / 90.0))

        raw_ratio = min(1.0, len(self.dairy_detections) / DAIRY_MAX)
        self.dairy_level += (raw_ratio - self.dairy_level) * 0.15
        draw_dairy_bar(display, self.dairy_level)
        draw_nudenet_panel(display, self.nude_detections, NudeNetAnalyzer.label_fr)
        if hand_points:
            draw_two_fingers_flash(display, hand_points)
        if self.hallelujah_started_at is not None:
            elapsed = now - self.hallelujah_started_at
            if elapsed > HALLELUJAH_DISPLAY_S:
                self.hallelujah_started_at = None
            else:
                draw_hallelujah(display, elapsed)

        self.video.set_frame(display)
        self.video.update()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.bias_window.close()
        event.accept()


def main():
    args = sys.argv[1:]
    camera_index = 0
    use_test = "--test" in args
    for i, a in enumerate(args):
        if a.startswith("--camera="):
            camera_index = int(a.split("=", 1)[1])
        elif a == "--camera" and i + 1 < len(args):
            camera_index = int(args[i + 1])
    app = QApplication(sys.argv)
    window = MainWindow(camera_index=camera_index, use_test_pattern=use_test)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
