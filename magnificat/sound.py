"""Hallelujah sonore : synthèse hardcodée, aucun fichier externe requis."""
from __future__ import annotations

import threading

import numpy as np


class HallelujahSound:
    """Joue un hallelujah synthétique (chœur d'orgue) via sounddevice."""

    NOTES = [
        (392.00, 0.0, 0.9),    # G4  —
        (392.00, 0.35, 0.9),   # G4  —
        (440.00, 0.70, 0.9),   # A4
        (349.23, 1.05, 0.9),   # F4
        (392.00, 1.40, 1.8),   # G4
        (349.23, 2.20, 0.9),   # F4
        (329.63, 2.55, 0.9),   # E4
        (293.66, 2.90, 0.9),   # D4
        (261.63, 3.25, 0.9),   # C4
        (293.66, 3.60, 2.4),   # D4
    ]

    def __init__(self):
        self._thread = None

    def play(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._play_sync, daemon=True)
        self._thread.start()

    def _play_sync(self):
        try:
            import sounddevice as sd
            fs = 44100
            total = 6.0
            t = np.arange(int(fs * total)) / fs
            audio = np.zeros_like(t)
            for freq, start, dur in self.NOTES:
                n0 = int(start * fs)
                n1 = min(int((start + dur) * fs), len(t))
                if n1 <= n0:
                    continue
                tt = t[n0:n1] - t[n0]
                env = np.minimum(1.0, tt / 0.05) * np.exp(-tt / dur * 2.5)
                tone = np.sin(2 * np.pi * freq * tt)
                tone += 0.4 * np.sin(2 * np.pi * freq * 2 * tt)
                tone += 0.2 * np.sin(2 * np.pi * freq * 3 * tt)
                audio[n0:n1] += tone * env
            audio *= 0.25 / max(1.0, np.max(np.abs(audio)))
            stereo = np.stack([audio, audio], axis=1)
            sd.play(stereo, fs, blocking=True)
        except Exception:
            pass
