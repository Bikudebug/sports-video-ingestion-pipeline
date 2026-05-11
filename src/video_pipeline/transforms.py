#!/usr/bin/env python3
"""
Professional Video Processing and Analysis Pipeline

This script performs robust video-source handling, embedded metadata extraction,
frame sampling, resolution normalization, RGB / grayscale frame export, frame
quality analysis, scene-boundary analysis, processed-video generation, preview
video generation, structured manifest writing, and output validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
    ".mts", ".m2ts", ".mpeg", ".mpg", ".m3u8"
}

DEFAULT_STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")

from .common import *
from .inputs import *


def open_video(input_source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {input_source}. "
            "Check path, codec support, network access, or stream accessibility."
        )

    return cap


def read_video_metadata_from_capture(
    cap: cv2.VideoCapture,
    input_source: str,
    config: Dict[str, Any],
    source_fps_fallback: float,
) -> Dict[str, Any]:
    fps_reported = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_used = fps_reported if fps_reported > 0 else float(source_fps_fallback)

    if frame_count > 0 and fps_used > 0:
        duration_sec = frame_count / fps_used
    else:
        duration_sec = None

    return {
        "source_path": input_source,
        "is_stream": is_stream_source(input_source, config),
        "extension": source_extension(input_source, config),
        "fps_reported": fps_reported,
        "fps_used": fps_used,
        "width": width,
        "height": height,
        "frame_count_reported": frame_count,
        "duration_sec_estimated": duration_sec,
    }


def compute_sampling_interval(fps_used: float, sampling_cfg: Dict[str, Any]) -> int:
    mode = str(sampling_cfg.get("mode", "target_fps")).lower()

    if mode == "target_fps":
        target_fps = float(sampling_cfg.get("target_fps", 5))

        if target_fps <= 0:
            raise ValueError("target_fps must be positive.")

        return max(1, int(round(fps_used / target_fps)))

    if mode == "every_n_frames":
        every_n_frames = int(sampling_cfg.get("every_n_frames", 10))

        if every_n_frames <= 0:
            raise ValueError("every_n_frames must be positive.")

        return every_n_frames

    raise ValueError(
        f"Unsupported sampling mode: {mode}. "
        "Use 'target_fps' or 'every_n_frames'."
    )


def resize_frame_bgr(frame_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    return cv2.resize(
        frame_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def gray_to_bgr(frame_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
