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
from .config import *
from .inputs import *
from .metadata import *
from .pipeline import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional Video Processing and Analysis Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input source. Overrides input.source_path in YAML.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output root directory. Overrides output.output_dir in YAML.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Use recursive folder search. Overrides input.recursive=true.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)

    input_source = args.input or config.get("input", {}).get("source_path")
    if input_source is None:
        print("[ERROR] No input source specified in YAML or --input.", file=sys.stderr)
        sys.exit(1)

    output_root = Path(
        args.output or
        config.get("output", {}).get("output_dir", "outputs/video_processing")
    )

    recursive = bool(config.get("input", {}).get("recursive", False)) or bool(args.recursive)

    try:
        inputs = collect_inputs(
            source_path=input_source,
            recursive=recursive,
            config=config,
        )
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)

        error_record = build_metadata_error_record(
            input_source=input_source,
            config=config,
            error_type=classify_error(str(exc)),
            message=str(exc),
        )

        error_path = output_root / "input_collection_error.json"
        write_json(error_record, error_path)

        print(f"[FAILED] Could not collect inputs: {exc}")
        print(f"[INFO] Error JSON saved: {error_path}")
        sys.exit(1)

    if not inputs:
        output_root.mkdir(parents=True, exist_ok=True)

        error_record = build_metadata_error_record(
            input_source=input_source,
            config=config,
            error_type="no_supported_video_files_found",
            message="The input folder did not contain supported video files.",
        )

        error_path = output_root / "no_supported_video_files_found.json"
        write_json(error_record, error_path)

        print("[FAILED] No supported video files found.")
        print(f"[INFO] Error JSON saved: {error_path}")
        sys.exit(1)

    print(f"[INFO] Total input source(s): {len(inputs)}")
    print(f"[INFO] Output root: {output_root}")

    pipeline_cfg = config.get("pipeline", {})
    continue_on_error = bool(pipeline_cfg.get("continue_on_error", True))

    all_summaries = []
    success_count = 0
    fail_count = 0

    for input_item in inputs:
        try:
            summary = process_one_source(
                input_source=input_item,
                config=config,
                output_root=output_root,
            )

            all_summaries.append(summary)

            if str(summary.get("final_status", "")).startswith("success"):
                success_count += 1
            else:
                fail_count += 1

        except Exception as exc:
            fail_count += 1

            error_summary = {
                "pipeline": "Professional Video Processing and Analysis Pipeline",
                "created_utc": utc_now_iso(),
                "input_source": input_item,
                "final_status": "failed_unhandled_exception",
                "error_type": classify_error(str(exc)),
                "message": short_error(str(exc)),
            }

            all_summaries.append(error_summary)

            print(f"[ERROR] Failed processing {input_item}")
            print(short_error(str(exc), limit=1000))

            if not continue_on_error:
                break

    batch_summary = {
        "pipeline": "Professional Video Processing and Analysis Pipeline",
        "created_utc": utc_now_iso(),
        "input_source_argument": input_source,
        "output_root": str(output_root),
        "num_inputs": len(inputs),
        "success_count": success_count,
        "fail_count": fail_count,
        "summaries": all_summaries,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = output_root / "batch_processing_summary.json"
    write_json(batch_summary, batch_summary_path)

    print("[DONE]")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Batch summary: {batch_summary_path}")

    if success_count == 0 and fail_count > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
