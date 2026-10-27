"""Analyseurs d'images : NudeNet, YOLO-World, MediaPipe (mains)."""
from __future__ import annotations

import threading

import cv2
import numpy as np


def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception:
        return None


class NudeNetAnalyzer:
    """Affiche toutes les infos que NudeNet peut obtenir (y compris les tétons)."""

    LABEL_FR = {
        "FEMALE_GENITALIA_COVERED": "génital féminin (couvert)",
        "FEMALE_GENITALIA_EXPOSED": "génital féminin (exposé)",
        "FEMALE_BREAST_COVERED": "sein féminin (couvert)",
        "FEMALE_BREAST_EXPOSED": "sein féminin (exposé)",
        "BUTTOCKS_COVERED": "fesses (couvertes)",
        "BUTTOCKS_EXPOSED": "fesses (exposées)",
        "ANUS_COVERED": "anus (couvert)",
        "ANUS_EXPOSED": "anus (exposé)",
        "FEMALE_NIPPLE": "téton",
        "MALE_NIPPLE": "téton (mâle)",
        "MALE_GENITALIA_COVERED": "génital mâle (couvert)",
        "MALE_GENITALIA_EXPOSED": "génital mâle (exposé)",
        "ARMPITS_COVERED": "aisselle (couverte)",
        "ARMPITS_EXPOSED": "aisselle (exposée)",
        "BELLY_COVERED": "ventre (couvert)",
        "BELLY_EXPOSED": "ventre (exposé)",
        "FEET_COVERED": "pieds (couverts)",
        "FEET_EXPOSED": "pieds (exposés)",
        "FACE_MALE": "visage (mâle)",
        "FACE_FEMALE": "visage (femme)",
        "HOTSPOTS": "zones sensibles",
        "SWIMWEAR": "maillot de bain",
        "MALE_BREAST_COVERED": "poitrine mâle (couverte)",
        "MALE_BREAST_EXPOSED": "poitrine mâle (exposée)",
        "CLOTHED_MALE": "habillé (mâle)",
        "CLOTHED_FEMALE": "habillée (femme)",
        "DRESS": "robe",
        "FOREIGN_OBJECT": "objet étranger",
        "NO_FACE": "sans visage",
        "REPRESENTATIVE_HOLE": "orifice",
        "LOWER_ABDOMEN": "bas-ventre",
        "LOWER_BACK": "bas du dos",
        "CHEST": "torse",
        "LEGS": "jambes",
        "ARMS": "bras",
    }

    def __init__(self, model_path=None):
        from nudenet import NudeDetector
        self._detector = NudeDetector(model_path) if model_path else NudeDetector()
        self._lock = threading.Lock()

    def detect(self, bgr):
        with self._lock:
            try:
                return self._detector.detect(bgr)
            except Exception:
                return []

    @classmethod
    def label_fr(cls, label):
        return cls.LABEL_FR.get(label, label)


# Classes trackées par YOLO-World pour le Miracle Lacté.
DAIRY_CLASSES = ["cow", "udder", "baby bottle", "nipple", "breast"]


class YoloWorldAnalyzer:
    """Repère vaches, pis, biberons et tétons — plus il en voit, plus le Lait divin monte.

    Le slider « Foi » (bias) permet de biaiser le modèle pour qu'il en voie partout :
    à 0, comportement normal ; à fond, chaque objet devient du Lait.
    """

    def __init__(self, classes=None, model_path=None):
        from ultralytics import YOLO
        self._model = YOLO(model_path or "yolov8s-worldv2")
        self._model.set_classes(classes or DAIRY_CLASSES)
        self._lock = threading.Lock()
        self.conf_threshold = 0.25

    def detect(self, bgr, bias=0.0):
        """Retourne (detections_dairy, detections_autres).

        bias (0..1) : au-delà, les détections normales (person, cup, etc.) sont
        converties en objets laitiers — le modèle « en voit partout ».
        """
        results = self._model.predict(bgr, conf=self.conf_threshold, verbose=False)
        detections = []
        others = []
        if not results:
            return detections, others
        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            if label in DAIRY_CLASSES and conf >= self.conf_threshold:
                detections.append((label, conf, xyxy))
            else:
                others.append((label, conf, xyxy))
        if bias > 0:
            for label, conf, xyxy in others:
                adjusted = conf * (1.0 + 5.0 * bias)
                if adjusted >= self.conf_threshold:
                    detections.append(("lait_divin", adjusted, xyxy))
        return detections, others


class FaceTracker:
    """MediaPipe Face Mesh : suit les têtes des comédiennes pour les auréoles."""

    def __init__(self):
        self._mp = _import_mediapipe()
        self._mesh = None
        if self._mp is not None:
            try:
                self._mesh = self._mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=6,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                )
            except Exception:
                self._mesh = None

    @property
    def available(self):
        return self._mesh is not None

    def head_boxes(self, bgr):
        """Retourne une boîte (x1, y1, x2, y2) par tête détectée."""
        if self._mesh is None:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        h, w = bgr.shape[:2]
        boxes = []
        for face in result.multi_face_landmarks:
            xs = [p.x * w for p in face.landmark]
            ys = [p.y * h for p in face.landmark]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes


class HandSignalAnalyzer:
    """MediaPipe Hands : deux doigts levés (index + majeur) déclenchent l'hallelujah."""

    GESTURE_COOLDOWN_S = 3.0

    def __init__(self):
        mp = _import_mediapipe()
        self._mp = mp
        self._hands = None
        if mp is not None:
            try:
                self._hands = mp.solutions.hands.Hands(
                    model_complexity=0,
                    max_num_hands=4,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._hands = None

    @property
    def available(self):
        return self._hands is not None

    def _distance(self, a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _fingers_up(self, lm, w, h):
        """Retourne une liste de booléens [pouce, index, majeur, annulaire, auriculaire]."""
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]
        for tip, pip in zip(tip_ids, pip_ids):
            fingers.append(pts[tip][1] < pts[pip][1])
        return pts, fingers

    def detect_two_fingers(self, bgr):
        """Retourne (main_detectee, deux_doigts_leves, points_de_contour)."""
        if self._hands is None:
            return False, False, []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return False, False, []
        h, w = bgr.shape[:2]
        any_two = False
        points = []
        for lm in result.multi_hand_landmarks:
            pts, fingers = self._fingers_up(lm, w, h)
            points = pts
            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                any_two = True
        return True, any_two, points
