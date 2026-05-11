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


def validate_outputs(
    video_output_dir: Path,
    metadata_path: Path,
    frame_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    manifest_paths: Dict[str, str],
    video_output_paths: Dict[str, str],
    analysis_paths: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    validation_cfg = config.get("output_validation", {})

    checks: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    def check(condition: bool, name: str, error_message: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append({
                "check": name,
                "message": error_message,
            })

    if bool(validation_cfg.get("validate_output_structure", True)):
        check(video_output_dir.exists(), "video_output_dir_exists", f"Missing {video_output_dir}")
        check((video_output_dir / "embedded_metadata").exists(), "embedded_metadata_dir_exists", "Missing embedded_metadata directory.")
        check(metadata_path.exists(), "metadata_json_exists", f"Missing metadata file: {metadata_path}")
        check((video_output_dir / "frames").exists(), "frames_dir_exists", "Missing frames directory.")
        check((video_output_dir / "manifests").exists(), "manifests_dir_exists", "Missing manifests directory.")
        check((video_output_dir / "logs").exists(), "logs_dir_exists", "Missing logs directory.")

    for name, path_str in manifest_paths.items():
        path = Path(path_str)
        check(path.exists(), f"{name}_exists", f"Missing manifest/report file: {path}")

    required_fields = [
        "source_video_id",
        "source_video_path",
        "frame_index",
        "sampled_frame_number",
        "timestamp_sec",
        "segment_id",
        "saved",
        "output_paths",
        "relative_paths",
        "inference_input",
        "processed_shape",
        "temporal_info",
        "transformations",
        "quality",
        "scene_detection",
    ]

    missing_required_field_count = 0

    if bool(validation_cfg.get("validate_required_manifest_fields", True)):
        for idx, record in enumerate(frame_records):
            for field in required_fields:
                if field not in record:
                    missing_required_field_count += 1
                    errors.append({
                        "check": "required_manifest_fields",
                        "message": f"Frame record {idx} missing field: {field}",
                    })

    checks["all_required_manifest_fields_present"] = missing_required_field_count == 0

    missing_frame_files = []

    if bool(validation_cfg.get("validate_saved_frame_paths", True)):
        for record in frame_records:
            if not record.get("saved", False):
                continue

            output_paths = record.get("output_paths", {})

            for key in ["rgb", "gray"]:
                path_str = output_paths.get(key)

                if path_str and not Path(path_str).exists():
                    missing_frame_files.append(path_str)

    checks["all_saved_frame_paths_exist"] = len(missing_frame_files) == 0

    for missing in missing_frame_files[:50]:
        errors.append({
            "check": "saved_frame_path_exists",
            "message": f"Missing saved frame file: {missing}",
        })

    saved_rgb_in_manifest = sum(
        1 for r in frame_records
        if r.get("saved", False) and r.get("output_paths", {}).get("rgb") is not None
    )

    saved_gray_in_manifest = sum(
        1 for r in frame_records
        if r.get("saved", False) and r.get("output_paths", {}).get("gray") is not None
    )

    sampled_frames_summary = summary.get("sampling", {}).get("sampled_frames")

    check(
        sampled_frames_summary == len(frame_records),
        "sampled_frame_count_matches_manifest",
        f"summary sampled_frames={sampled_frames_summary}, manifest rows={len(frame_records)}",
    )

    saved_frames_summary = summary.get("saved_frames", {})

    expected_rgb = (
        int(saved_frames_summary.get("good_rgb_frames", 0)) +
        int(saved_frames_summary.get("poor_rgb_frames", 0))
    )

    expected_gray = (
        int(saved_frames_summary.get("good_gray_frames", 0)) +
        int(saved_frames_summary.get("poor_gray_frames", 0))
    )

    check(
        expected_rgb == saved_rgb_in_manifest,
        "saved_rgb_count_matches_manifest",
        f"summary saved rgb={expected_rgb}, manifest saved rgb={saved_rgb_in_manifest}",
    )

    check(
        expected_gray == saved_gray_in_manifest,
        "saved_gray_count_matches_manifest",
        f"summary saved gray={expected_gray}, manifest saved gray={saved_gray_in_manifest}",
    )

    if bool(validation_cfg.get("validate_video_and_analysis_outputs", True)):
        for name, path_str in video_output_paths.items():
            if path_str:
                check(Path(path_str).exists(), f"{name}_exists", f"Missing video output: {path_str}")

        for name, path_str in analysis_paths.items():
            if path_str:
                check(Path(path_str).exists(), f"{name}_exists", f"Missing analysis output: {path_str}")

    status = "PASS" if len(errors) == 0 else "FAIL"

    return {
        "report_type": "output_validation",
        "created_utc": utc_now_iso(),
        "status": status,
        "video_output_dir": str(video_output_dir),
        "num_manifest_rows": len(frame_records),
        "num_missing_frame_files": len(missing_frame_files),
        "num_errors": len(errors),
        "checks": checks,
        "errors": errors[:200],
        "interpretation": {
            "PASS": "The output hierarchy, frame paths, manifest fields, and enabled artefacts are consistent.",
            "FAIL": "Some required output files, frame paths, counts, or metadata fields are missing or inconsistent.",
        },
    }
