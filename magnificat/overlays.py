"""Superpositions visuelles : auréoles dorées et inscription SANCTA au-dessus des têtes."""
from __future__ import annotations

import cv2
import numpy as np


def draw_halo(display, original, head_box):
    """Auréole dorée autour d'une tête.

    L'arrière de l'anneau est peint en premier, puis la région du crâne est
    recopiée depuis l'image originale par-dessus : l'arrière de l'auréole
    passe donc derrière le crâne. L'avant est ensuite redessiné au premier plan.
    """
    hx1, hy1, hx2, hy2 = [int(v) for v in head_box]
    bw = max(1, hx2 - hx1)
    bh = max(1, hy2 - hy1)
    cx = (hx1 + hx2) // 2
    r = max(24, int(bw * 0.62))
    cy = int(hy1 - bh * 0.15)

    gold = (72, 215, 255)
    rear = display.copy()
    cv2.ellipse(rear, (cx, cy), (r, r), 0, 160, 380, gold, 3, cv2.LINE_AA)
    cv2.addWeighted(rear, 0.55, display, 0.45, 0, display)

    mask = np.zeros(display.shape[:2], np.uint8)
    cv2.ellipse(
        mask,
        (cx, int(hy1 + bh * 0.45)),
        (int(bw * 0.75), int(bh * 0.85)),
        0, 0, 360, 255, -1,
    )
    display[:] = np.where(mask[:, :, None] == 255, original, display)

    glow = display.copy()
    cv2.ellipse(glow, (cx, cy), (r + 5, r + 5), 0, 0, 360, gold, 1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, display, 0.65, 0, display)
    cv2.ellipse(display, (cx, cy), (r, r), 0, 340, 200, gold, 3, cv2.LINE_AA)
    return cx, cy, r


def draw_sancta(canvas_bgr, cx, cy, scale=1.0):
    """Inscription SANCTA au-dessus de la tête."""
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "SANCTA"
    thickness = max(1, int(round(2 * scale)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = cx - tw // 2
    y = cy
    cv2.putText(canvas_bgr, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, text, (x, y), font, scale, (72, 215, 255), thickness, cv2.LINE_AA)


def draw_bbox_label(canvas_bgr, x1, y1, x2, y2, label, score=None, color=(0, 255, 255)):
    """Boîte + étiquette."""
    cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 2)
    text = label if score is None else f"{label} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    cv2.rectangle(canvas_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 0, 0), -1)
    cv2.putText(canvas_bgr, text, (x1 + 2, max(11, y1 - 6)), font, 0.5, color, 1, cv2.LINE_AA)


def draw_dairy_bar(canvas_bgr, ratio, label="LAIT DIVIN"):
    """Barre de « Lait divin » en haut de l'écran."""
    h, w = canvas_bgr.shape[:2]
    bar_w = int(w * 0.6)
    bar_h = 26
    x = (w - bar_w) // 2
    y = 8
    cv2.rectangle(canvas_bgr, (x, y), (x + bar_w, y + bar_h), (30, 30, 30), -1)
    cv2.rectangle(canvas_bgr, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)
    fill = int(bar_w * max(0.0, min(1.0, ratio)))
    if fill > 0:
        cv2.rectangle(canvas_bgr, (x + 2, y + 2), (x + 2 + fill, y + bar_h - 2), (200, 230, 255), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    text = f"{label} {ratio * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.putText(canvas_bgr, text, (x + bar_w // 2 - tw // 2, y + bar_h - 8), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, text, (x + bar_w // 2 - tw // 2, y + bar_h - 8), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_nudenet_panel(canvas_bgr, detections, label_map):
    """Panneau latéral : toutes les infos que NudeNet peut obtenir."""
    h, w = canvas_bgr.shape[:2]
    panel_w = min(300, w // 3)
    overlay = canvas_bgr.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, canvas_bgr, 0.45, 0, canvas_bgr)
    cv2.line(canvas_bgr, (w - panel_w, 0), (w - panel_w, h), (72, 215, 255), 1)
    title = "Nudenet : infos"
    cv2.putText(canvas_bgr, title, (w - panel_w + 10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, (72, 215, 255), 1, cv2.LINE_AA)
    y = 44
    if not detections:
        cv2.putText(canvas_bgr, "aucune detection", (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        return
    for det in detections[:14]:
        label = label_map(det.get("label", ""))
        score = det.get("score", 0.0)
        text = f"{label} {score:.2f}"
        cv2.putText(canvas_bgr, text, (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
        if y > h - 12:
            break


def draw_two_fingers_flash(canvas_bgr, points):
    """Surligne la main qui fait le signe des deux doigts."""
    if not points:
        return
    for pt in points:
        cv2.circle(canvas_bgr, pt, 3, (0, 255, 255), -1)
    cv2.polylines(canvas_bgr, [np.array(points, np.int32)], False, (0, 255, 255), 1)


def draw_hallelujah(canvas_bgr, elapsed):
    """Grand « HALLELUJAH » plein écran pendant l'extase sonore."""
    if elapsed is None or elapsed < 0:
        return
    h, w = canvas_bgr.shape[:2]
    scale = 1.0 + min(elapsed, 1.0) * 0.6
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "HALLELUJAH"
    (tw, th), _ = cv2.getTextSize(text, font, scale * 3, 6)
    x = (w - tw) // 2
    y = (h + th) // 2
    alpha = max(0.0, 1.0 - elapsed / 2.0)
    if alpha <= 0:
        return
    overlay = canvas_bgr.copy()
    cv2.putText(overlay, text, (x, y), font, scale * 3, (72, 215, 255), 6, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.8, canvas_bgr, 1 - alpha * 0.8, 0, canvas_bgr)


def draw_bias_indicators(canvas_bgr, bias, detections):
    """Souligne les détections (réelles ou « révélées » par la Foi)."""
    colors = {
        "cow": (255, 200, 100),
        "udder": (255, 180, 200),
        "baby bottle": (200, 255, 180),
        "nipple": (180, 180, 255),
        "breast": (255, 220, 180),
        "lait_divin": (200, 230, 255),
    }
    for label, conf, (x1, y1, x2, y2) in detections:
        color = colors.get(label, (200, 230, 255))
        if label == "lait_divin":
            draw_bbox_label(canvas_bgr, int(x1), int(y1), int(x2), int(y2), "lait divin", conf, color)
        else:
            draw_bbox_label(canvas_bgr, int(x1), int(y1), int(x2), int(y2), label, conf, color)
