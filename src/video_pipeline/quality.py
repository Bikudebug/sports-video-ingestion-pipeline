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


def compute_quality_scores(frame_gray: np.ndarray, quality_cfg: Dict[str, Any]) -> Dict[str, Any]:
    blur_score = float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
    brightness_score = float(np.mean(frame_gray))
    contrast_score = float(np.std(frame_gray))

    blur_threshold = float(quality_cfg.get("blur_threshold", 80.0))
    brightness_min = float(quality_cfg.get("brightness_min", 35.0))
    brightness_max = float(quality_cfg.get("brightness_max", 225.0))
    contrast_min = float(quality_cfg.get("contrast_min", 10.0))

    use_blur_check = bool(quality_cfg.get("use_blur_check", True))
    use_brightness_check = bool(quality_cfg.get("use_brightness_check", True))
    use_contrast_check = bool(quality_cfg.get("use_contrast_check", True))

    is_blurry = blur_score < blur_threshold if use_blur_check else False
    is_too_dark = brightness_score < brightness_min if use_brightness_check else False
    is_too_bright = brightness_score > brightness_max if use_brightness_check else False
    is_low_contrast = contrast_score < contrast_min if use_contrast_check else False

    is_good_frame = not (
        is_blurry or
        is_too_dark or
        is_too_bright or
        is_low_contrast
    )

    return {
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "blur_threshold": blur_threshold,
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
        "contrast_min": contrast_min,
        "is_blurry": bool(is_blurry),
        "is_too_dark": bool(is_too_dark),
        "is_too_bright": bool(is_too_bright),
        "is_low_contrast": bool(is_low_contrast),
        "is_good_frame": bool(is_good_frame),
    }


def create_quality_report(
    frame_records: List[Dict[str, Any]],
    quality_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    blur_scores = []
    brightness_scores = []
    contrast_scores = []

    good_count = 0
    poor_count = 0
    blurry_count = 0
    too_dark_count = 0
    too_bright_count = 0
    low_contrast_count = 0

    for record in frame_records:
        q = record["quality"]

        blur_scores.append(float(q["blur_score"]))
        brightness_scores.append(float(q["brightness_score"]))
        contrast_scores.append(float(q["contrast_score"]))

        if q["is_good_frame"]:
            good_count += 1
        else:
            poor_count += 1

        if q["is_blurry"]:
            blurry_count += 1

        if q["is_too_dark"]:
            too_dark_count += 1

        if q["is_too_bright"]:
            too_bright_count += 1

        if q["is_low_contrast"]:
            low_contrast_count += 1

    return {
        "report_type": "frame_quality_analysis",
        "total_sampled_frames": len(frame_records),
        "good_frames": good_count,
        "poor_frames": poor_count,
        "failure_counts": {
            "blurry": blurry_count,
            "too_dark": too_dark_count,
            "too_bright": too_bright_count,
            "low_contrast": low_contrast_count,
        },
        "thresholds_used": {
            "blur_threshold": float(quality_cfg.get("blur_threshold", 80.0)),
            "brightness_min": float(quality_cfg.get("brightness_min", 35.0)),
            "brightness_max": float(quality_cfg.get("brightness_max", 225.0)),
            "contrast_min": float(quality_cfg.get("contrast_min", 10.0)),
        },
        "score_statistics": {
            "blur_score": numeric_summary(blur_scores),
            "brightness_score": numeric_summary(brightness_scores),
            "contrast_score": numeric_summary(contrast_scores),
        },
        "interpretation": {
            "blur_score": "Higher is sharper; lower means more blur.",
            "brightness_score": "Mean grayscale intensity in [0, 255]. Very low is dark; very high is overexposed.",
            "contrast_score": "Standard deviation of grayscale intensity. Lower values indicate flatter, low-contrast frames.",
            "note": "Thresholds are empirical and configurable. Calibrate them per dataset.",
        },
    }
