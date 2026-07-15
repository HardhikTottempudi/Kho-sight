"""Audio "Kho" keyword spotting (rule 4.2.1: the call must be loud and audible).

M0 ships a shout-detector stub: a short-window RMS spike in the speech band is
treated as a candidate call, used only to *timestamp* khos that vision already
sees as a sit/stand swap (never to create events alone).

M1 replaces `is_call` with a small keyword-spotting model (1D-CNN / conformer on
log-mel spectrograms) trained on "kho" clips mined from your match audio — see
scripts/prepare_dataset.py --audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KhoSpotter:
    sample_rate: int = 16000
    window_s: float = 0.25
    spike_ratio: float = 3.0  # window RMS vs rolling background RMS

    _background_rms: float = 0.0
    _alpha: float = 0.02

    def is_call(self, samples: np.ndarray) -> bool:
        """samples: mono float window aligned to the current video frame."""
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        bg = self._background_rms
        self._background_rms = (1 - self._alpha) * bg + self._alpha * rms
        if bg <= 1e-8:
            return False
        return rms > self.spike_ratio * bg
