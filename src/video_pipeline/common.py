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



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout} seconds."
    except Exception as exc:
        return 1, "", str(exc)


def resolve_media_tool(binary_name: str) -> Optional[str]:
    resolved = shutil.which(binary_name)
    if resolved is not None:
        return resolved

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            matches = list(winget_packages.glob(f"**/{binary_name}.exe"))
            if matches:
                return str(matches[0])

    return None


def safe_int(value: Any) -> Optional[int]:
    if value is None or value == "N/A":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_ratio(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    value = str(value).strip()
    if value in {"", "0/0", "N/A"}:
        return None

    try:
        if "/" in value:
            return float(Fraction(value))
        return float(value)
    except Exception:
        return None


def parse_time_base_seconds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(Fraction(str(value)))
    except Exception:
        return None


def clean_float(value: Optional[float], digits: int = 6) -> Any:
    if value is None:
        return "not_available"
    return round(float(value), digits)


def choose_first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def classify_error(message: str) -> str:
    msg = (message or "").lower()

    if "no such file" in msg or "cannot find the file" in msg or "input not found" in msg:
        return "file_not_found"

    if "moov atom not found" in msg:
        return "truncated_or_invalid_mp4"

    if "invalid data found" in msg:
        return "invalid_video_file"

    if "end of file" in msg or "partial file" in msg or "truncated" in msg:
        return "truncated_or_partially_downloaded_file"

    if "no video stream" in msg:
        return "no_video_stream_found"

    if "missing hls segment" in msg:
        return "broken_hls_missing_segment"

    if "could not open video" in msg:
        return "opencv_decode_failure"

    if "timed out" in msg:
        return "timeout"

    if "unsupported" in msg:
        return "unsupported_input"

    return "video_read_error"


def short_error(message: str, limit: int = 1200) -> str:
    message = (message or "").strip()
    if len(message) <= limit:
        return message
    return message[-limit:]


def numeric_summary(values: List[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    arr = np.array(values, dtype=np.float64)

    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def fps_label(fps: float) -> str:
    if abs(fps - int(fps)) < 1e-6:
        return f"{int(fps)}fps"
    return f"{str(round(fps, 3)).replace('.', 'p')}fps"


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_relative_path(path_str: Optional[str], base_dir: Path) -> Optional[str]:
    if path_str is None:
        return None

    try:
        return str(Path(path_str).resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return path_str
