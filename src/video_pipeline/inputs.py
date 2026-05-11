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


def configured_stream_prefixes(config: Dict[str, Any]) -> Tuple[str, ...]:
    prefixes = config.get("input", {}).get("stream_prefixes", list(DEFAULT_STREAM_PREFIXES))
    return tuple(str(p).lower() for p in prefixes)


def is_stream_source(source: str, config: Optional[Dict[str, Any]] = None) -> bool:
    prefixes = configured_stream_prefixes(config or {})
    return str(source).lower().startswith(prefixes)


def source_extension(source: str, config: Optional[Dict[str, Any]] = None) -> str:
    if is_stream_source(source, config):
        parsed = urlparse(source)
        return Path(parsed.path).suffix.lower()
    return Path(source).suffix.lower()


def source_name(source: str, config: Optional[Dict[str, Any]] = None) -> str:
    if is_stream_source(source, config):
        parsed = urlparse(source)
        name = Path(parsed.path).name
        return name if name else "stream_source"
    return Path(source).name


def source_stem(source: str, config: Optional[Dict[str, Any]] = None) -> str:
    if is_stream_source(source, config):
        parsed = urlparse(source)
        stem = Path(parsed.path).stem
        return stem if stem else "stream_source"
    return Path(source).stem


def safe_video_id(source: str, config: Dict[str, Any], append_hash: bool = False) -> str:
    stem = source_stem(source, config)
    stem = stem.replace(" ", "_")

    if is_stream_source(source, config) or append_hash:
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{digest}"

    return stem


def validate_local_hls_playlist(
    input_path: str,
    config: Dict[str, Any],
    validate_segments: bool = True,
) -> None:
    path = Path(input_path)

    if path.suffix.lower() != ".m3u8":
        return

    if is_stream_source(input_path, config):
        return

    if not path.exists():
        raise RuntimeError(f"Input not found: {input_path}")

    if not validate_segments:
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    base_dir = path.parent
    missing_segments = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if line.startswith(("http://", "https://")):
            continue

        segment_path = (base_dir / line).resolve()

        if not segment_path.exists():
            missing_segments.append(str(segment_path))

    if missing_segments:
        joined = "\n".join(missing_segments)
        raise RuntimeError(f"Missing HLS segment file(s):\n{joined}")


def is_supported_source(source: str, config: Dict[str, Any]) -> bool:
    input_cfg = config.get("input", {})
    supported_extensions = set(input_cfg.get("supported_extensions", sorted(DEFAULT_VIDEO_EXTENSIONS)))
    allow_unknown_url_extension = bool(input_cfg.get("allow_unknown_url_extension", True))

    if is_stream_source(source, config):
        ext = source_extension(source, config)

        if source.lower().startswith(("rtsp://", "rtmp://")):
            return True

        if ext in supported_extensions:
            return True

        if allow_unknown_url_extension:
            return True

        return False

    return Path(source).suffix.lower() in supported_extensions


def collect_inputs(source_path: str, recursive: bool, config: Dict[str, Any]) -> List[str]:
    if is_stream_source(source_path, config):
        if not is_supported_source(source_path, config):
            raise FileNotFoundError(f"Unsupported stream or URL input: {source_path}")
        return [source_path]

    path = Path(source_path)

    if path.is_file():
        if not is_supported_source(source_path, config):
            raise FileNotFoundError(f"Unsupported video extension: {path.suffix}")
        return [str(path)]

    if path.is_dir():
        pattern = "**/*" if recursive else "*"

        files = [
            str(p)
            for p in path.glob(pattern)
            if p.is_file() and is_supported_source(str(p), config)
        ]

        return sorted(files)

    raise FileNotFoundError(f"Input not found: {source_path}")
