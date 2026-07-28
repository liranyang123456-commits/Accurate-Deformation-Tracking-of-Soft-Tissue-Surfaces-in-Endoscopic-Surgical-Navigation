"""Read-only runtime hooks for frozen tracker response tensors.

OpenCV TrackerNano exposes only bbox and scalar score. The frozen Scheme-C
implementation does, however, construct a real Nano Gaussian prior map before
hierarchical matching. This context manager captures that map and any ndarray
responses returned by the HistAB/Contour hierarchy without editing frozen code.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import numpy as np


class InternalResponseCapture(AbstractContextManager):
    def __init__(self, tracker: Any):
        self.tracker = tracker
        self.nano_prior_map: np.ndarray | None = None
        self.hierarchy_arrays: dict[str, np.ndarray] = {}
        self._restore: list[tuple[Any, str, Any]] = []

    def _patch(self, owner: Any, name: str, wrapped: Any) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return
        self._restore.append((owner, name, original))
        setattr(owner, name, wrapped(original))

    def __enter__(self) -> "InternalResponseCapture":
        scheme = self.tracker._scheme_mod()
        deformation = self.tracker._defm

        def wrap_prior(original):
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                prior = result[0] if isinstance(result, tuple) else result
                if isinstance(prior, np.ndarray):
                    self.nano_prior_map = np.asarray(prior).copy()
                return result
            return call

        def wrap_hierarchy(original):
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            self.hierarchy_arrays[str(key)] = np.asarray(value).copy()
                return result
            return call

        self._patch(scheme, "get_or_build_nano_prior_map", wrap_prior)
        self._patch(deformation, "find_best_match_hierarchical", wrap_hierarchy)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()
        return False

    def update(self, frame: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        with self:
            output = self.tracker.update(frame)
        responses = {
            "nano_prior_map": self.nano_prior_map,
            "hierarchy_arrays": self.hierarchy_arrays,
            "opencv_nanotrack_dense_map_available": False,
        }
        return output, responses
