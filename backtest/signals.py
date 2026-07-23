"""Directional confidence signal, matching the live bot's get_kronos_confidence semantics:
fraction of forecast samples above the current price at day+1 and day+3 (index 0 and 2
of a 5-step forecast). Callers must only pass closes up to and including "today" — the
engine is responsible for not leaking future bars into this function.
"""

from typing import Optional, Tuple

import numpy as np


class ChronosSignal:
    """Wraps amazon/chronos-t5-small. Requires `torch` + `chronos-forecasting` installed
    and network/HuggingFace access to download weights on first use."""

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        import torch
        from chronos import ChronosPipeline
        self._torch = torch
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32
        )

    def get_confidence(self, closes: np.ndarray, horizon: int = 5,
                        num_samples: int = 100) -> Optional[Tuple[float, float]]:
        if len(closes) < 30:
            return None
        current = closes[-1]
        context = self._torch.tensor(closes, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            forecast = self.pipeline.predict(context, prediction_length=horizon,
                                              num_samples=num_samples)
        samples = forecast[0].numpy()
        conf_1d = float(np.mean(samples[:, 0] > current))
        conf_3d = float(np.mean(samples[:, min(2, horizon - 1)] > current))
        return conf_1d, conf_3d


class FallbackSignal:
    """NOT a trading signal — a deterministic stand-in used only when Chronos isn't
    available (no torch/model access), so the backtest engine can still be exercised
    end-to-end. It derives a pseudo-confidence from recent realized momentum. Do not
    use its output to judge whether the real strategy is profitable.
    """

    def get_confidence(self, closes: np.ndarray, horizon: int = 5,
                        num_samples: int = 100) -> Optional[Tuple[float, float]]:
        if len(closes) < 30:
            return None
        window = closes[-10:]
        momentum = (window[-1] - window[0]) / window[0]
        # squash momentum into a pseudo-probability around 0.5
        conf = 0.5 + np.clip(momentum * 2.0, -0.45, 0.45)
        return float(conf), float(conf)


def load_signal(use_chronos: bool):
    if use_chronos:
        return ChronosSignal()
    return FallbackSignal()
