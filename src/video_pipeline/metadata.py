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


def build_metadata_error_record(
    input_source: str,
    config: Dict[str, Any],
    error_type: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "metadata_schema": "video_metadata_with_file_health",
        "schema_version": "1.0",
        "created_utc": utc_now_iso(),
        "status": "failed",
        "input": {
            "source_path": input_source,
            "file_name": source_name(input_source, config),
            "file_extension": source_extension(input_source, config),
            "is_stream": is_stream_source(input_source, config),
        },
        "file_health": {
            "readable": False,
            "error_type": error_type,
            "message": short_error(message),
        },
        "container": "not_available",
        "codec_parameters": "not_available",
        "frame_rate": "not_available",
        "resolution": "not_available",
        "duration": "not_available",
        "frame_count": "not_available",
        "timestamp_information": "not_available",
        "gps_or_camera_identifier_metadata": "not_available",
        "required_fields_check": {
            "codec_parameters_available": False,
            "frame_rate_available": False,
            "resolution_available": False,
            "duration_available": False,
            "frame_count_available": False,
            "timestamp_information_available": False,
            "gps_or_camera_identifier_checked": False,
            "input_handled_gracefully": True,
        },
    }


def run_ffprobe(input_source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    metadata_cfg = config.get("metadata", {})
    ffprobe_path = resolve_media_tool("ffprobe")

    if ffprobe_path is None:
        raise RuntimeError("ffprobe not found. Install FFmpeg and add it to PATH.")

    validate_local_hls_playlist(
        input_path=input_source,
        config=config,
        validate_segments=bool(metadata_cfg.get("validate_local_hls_segments", True)),
    )

    command = [
        ffprobe_path,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_source,
    ]

    timeout = int(metadata_cfg.get("ffprobe_timeout_sec", 300))
    return_code, stdout, stderr = run_command(command, timeout=timeout)

    if return_code != 0:
        error_type = classify_error(stderr)
        raise RuntimeError(f"{error_type}: ffprobe failed for {input_source}\n{stderr}")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_ffprobe_json: ffprobe output was not valid JSON: {exc}") from exc

    return data


def ffprobe_count_frames(input_source: str, config: Dict[str, Any]) -> Optional[int]:
    metadata_cfg = config.get("metadata", {})
    ffprobe_path = resolve_media_tool("ffprobe")

    if ffprobe_path is None:
        return None

    if not bool(metadata_cfg.get("count_frames_with_ffprobe", True)):
        return None

    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "json",
        input_source,
    ]

    timeout = int(metadata_cfg.get("ffprobe_timeout_sec", 300))
    return_code, stdout, _stderr = run_command(command, timeout=timeout)

    if return_code != 0:
        return None

    try:
        data = json.loads(stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        return safe_int(streams[0].get("nb_read_frames"))
    except Exception:
        return None


def validate_full_decode(input_source: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    metadata_cfg = config.get("metadata", {})
    ffmpeg_path = resolve_media_tool("ffmpeg")

    if ffmpeg_path is None:
        return False, "ffmpeg not found. Install FFmpeg and add it to PATH."

    validate_local_hls_playlist(
        input_path=input_source,
        config=config,
        validate_segments=bool(metadata_cfg.get("validate_local_hls_segments", True)),
    )

    command = [
        ffmpeg_path,
        "-v", "error",
        "-i", input_source,
        "-f", "null",
        "-"
    ]

    timeout = int(metadata_cfg.get("decode_timeout_sec", 900))
    return_code, _stdout, stderr = run_command(command, timeout=timeout)

    if return_code == 0:
        return True, ""

    return False, stderr


def find_video_stream(ffprobe_data: Dict[str, Any]) -> Dict[str, Any]:
    for stream in ffprobe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream

    raise RuntimeError("no_video_stream_found: No video stream found in the input.")


def read_opencv_metadata(input_source: str) -> Dict[str, Optional[float]]:
    data: Dict[str, Optional[float]] = {
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
        "first_frame_readable": False,
    }

    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        return data

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    ok, frame = cap.read()
    data["first_frame_readable"] = bool(ok and frame is not None)

    cap.release()

    width_i = int(width) if width and width > 0 else None
    height_i = int(height) if height and height > 0 else None
    fps_f = float(fps) if fps and fps > 0 else None
    frame_count_i = int(frame_count) if frame_count and frame_count > 0 else None

    duration = None
    if frame_count_i is not None and fps_f is not None and fps_f > 0:
        duration = frame_count_i / fps_f

    data["width"] = width_i
    data["height"] = height_i
    data["fps"] = fps_f
    data["frame_count"] = frame_count_i
    data["duration_seconds"] = duration

    return data


def infer_representation(
    input_source: str,
    config: Dict[str, Any],
    codec_name: Optional[str],
) -> str:
    ext = source_extension(input_source, config)

    codec_map = {
        "h264": "H.264",
        "hevc": "H.265/HEVC",
        "h265": "H.265/HEVC",
        "mjpeg": "MJPEG",
        "mpeg4": "MPEG-4",
        "vp9": "VP9",
        "av1": "AV1",
    }

    codec = codec_map.get(str(codec_name).lower(), codec_name)

    if is_stream_source(input_source, config):
        if ext == ".m3u8":
            return f"Remote HLS / {codec}"
        return f"Remote stream / {codec}"

    if ext == ".m3u8":
        return f"Local HLS / {codec}"

    if ext == ".mp4":
        return f"MP4 / {codec}"

    if ext == ".mkv":
        return f"MKV / {codec}"

    if ext == ".avi":
        return f"AVI / {codec}"

    return f"{ext.replace('.', '').upper()} / {codec}" if ext else f"Video / {codec}"


def extract_gps_camera_metadata(
    format_tags: Dict[str, Any],
    video_tags: Dict[str, Any],
) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    combined.update(format_tags or {})
    combined.update(video_tags or {})

    keys = [
        "location",
        "LOCATION",
        "com.apple.quicktime.location.ISO6709",
        "gps",
        "GPS",
        "GPSLatitude",
        "GPSLongitude",
        "make",
        "Make",
        "model",
        "Model",
        "camera",
        "Camera",
        "camera_id",
        "camera-id",
        "device_id",
        "device-id",
        "source_id",
        "source-id",
        "creation_time",
        "date",
        "DATE",
    ]

    found = {}

    for key in keys:
        if key in combined:
            found[key] = combined[key]

    if found:
        return {
            "available": True,
            "metadata": found,
        }

    return {
        "available": False,
        "metadata": "not_available",
    }


def build_source_metadata(input_source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    metadata_cfg = config.get("metadata", {})

    ffprobe_data = run_ffprobe(input_source, config=config)
    fmt = ffprobe_data.get("format", {})
    video = find_video_stream(ffprobe_data)

    cv_meta = read_opencv_metadata(input_source)

    if not cv_meta["first_frame_readable"]:
        raise RuntimeError("opencv_decode_failure: OpenCV could not decode the first video frame.")

    if bool(metadata_cfg.get("validate_decode", False)):
        decode_ok, decode_error = validate_full_decode(input_source, config=config)
        if not decode_ok:
            error_type = classify_error(decode_error)
            raise RuntimeError(f"{error_type}: Full decode validation failed.\n{decode_error}")

    container_format = fmt.get("format_name")
    container_long_name = fmt.get("format_long_name")
    codec_name = video.get("codec_name")

    width = choose_first(safe_int(video.get("width")), cv_meta["width"])
    height = choose_first(safe_int(video.get("height")), cv_meta["height"])

    avg_fps_raw = video.get("avg_frame_rate")
    real_fps_raw = video.get("r_frame_rate")

    avg_fps = parse_ratio(avg_fps_raw)
    real_fps = parse_ratio(real_fps_raw)
    fps = choose_first(avg_fps, real_fps, cv_meta["fps"])

    container_duration = safe_float(fmt.get("duration"))
    video_duration = safe_float(video.get("duration"))
    duration_seconds = choose_first(video_duration, container_duration, cv_meta["duration_seconds"])

    ffprobe_frame_count = safe_int(video.get("nb_frames"))
    counted_frame_count = ffprobe_count_frames(input_source, config=config)
    opencv_frame_count = cv_meta["frame_count"]

    estimated_frame_count = None
    if duration_seconds is not None and fps is not None and fps > 0:
        estimated_frame_count = int(round(duration_seconds * fps))

    frame_count = choose_first(
        ffprobe_frame_count,
        counted_frame_count,
        opencv_frame_count,
        estimated_frame_count,
    )

    time_base_raw = video.get("time_base")
    time_base_seconds = parse_time_base_seconds(time_base_raw)

    if time_base_seconds is None and fps is not None and fps > 0:
        time_base_seconds = 1.0 / fps
        time_base_raw = f"1/{fps:.6f}"

    start_pts = choose_first(safe_int(video.get("start_pts")), 0)
    start_time_seconds = choose_first(safe_float(video.get("start_time")), 0.0)

    duration_ts_reported = safe_int(video.get("duration_ts"))

    duration_ts_computed = None
    if duration_seconds is not None and time_base_seconds is not None and time_base_seconds > 0:
        duration_ts_computed = int(round(duration_seconds / time_base_seconds))
    elif frame_count is not None:
        duration_ts_computed = frame_count

    duration_ts = choose_first(duration_ts_reported, duration_ts_computed, frame_count)

    gps_camera_metadata = extract_gps_camera_metadata(
        format_tags=fmt.get("tags", {}) or {},
        video_tags=video.get("tags", {}) or {},
    )

    record = {
        "metadata_schema": "video_metadata_with_file_health",
        "schema_version": "1.0",
        "created_utc": utc_now_iso(),
        "status": "success",
        "input": {
            "source_path": input_source,
            "file_name": source_name(input_source, config),
            "file_extension": source_extension(input_source, config),
            "is_stream": is_stream_source(input_source, config),
            "input_representation": infer_representation(
                input_source=input_source,
                config=config,
                codec_name=codec_name,
            ),
        },
        "file_health": {
            "readable": True,
            "error_type": "none",
            "message": "Input video was successfully opened, decoded, and metadata was extracted.",
        },
        "container": {
            "format_name": container_format if container_format is not None else "not_available",
            "format_long_name": container_long_name if container_long_name is not None else "not_available",
        },
        "codec_parameters": {
            "codec_name": codec_name if codec_name is not None else "not_available",
            "codec_long_name": video.get("codec_long_name") if video.get("codec_long_name") is not None else "not_available",
            "codec_profile": video.get("profile") if video.get("profile") is not None else "not_available",
            "codec_tag_string": video.get("codec_tag_string") if video.get("codec_tag_string") is not None else "not_available",
            "codec_tag": video.get("codec_tag") if video.get("codec_tag") is not None else "not_available",
            "pixel_format": video.get("pix_fmt") if video.get("pix_fmt") is not None else "not_available",
            "bits_per_raw_sample": safe_int(video.get("bits_per_raw_sample")) if safe_int(video.get("bits_per_raw_sample")) is not None else "not_available",
            "has_b_frames": safe_int(video.get("has_b_frames")) if safe_int(video.get("has_b_frames")) is not None else "not_available",
            "field_order": video.get("field_order") if video.get("field_order") is not None else "not_available",
            "level": safe_int(video.get("level")) if safe_int(video.get("level")) is not None else "not_available",
        },
        "frame_rate": {
            "fps": clean_float(fps),
            "average_frame_rate_raw": avg_fps_raw if avg_fps_raw is not None else "not_available",
            "average_frame_rate_fps": clean_float(avg_fps),
            "real_frame_rate_raw": real_fps_raw if real_fps_raw is not None else "not_available",
            "real_frame_rate_fps": clean_float(real_fps),
        },
        "resolution": {
            "width": width if width is not None else "not_available",
            "height": height if height is not None else "not_available",
            "resolution_string": f"{width}x{height}" if width is not None and height is not None else "not_available",
            "coded_width": safe_int(video.get("coded_width")) if safe_int(video.get("coded_width")) is not None else "not_available",
            "coded_height": safe_int(video.get("coded_height")) if safe_int(video.get("coded_height")) is not None else "not_available",
            "sample_aspect_ratio": video.get("sample_aspect_ratio") if video.get("sample_aspect_ratio") is not None else "not_available",
            "display_aspect_ratio": video.get("display_aspect_ratio") if video.get("display_aspect_ratio") is not None else "not_available",
        },
        "duration": {
            "duration_seconds": clean_float(duration_seconds),
            "duration_milliseconds": int(round(duration_seconds * 1000)) if duration_seconds is not None else "not_available",
        },
        "frame_count": {
            "number_of_frames": frame_count if frame_count is not None else "not_available",
        },
        "timestamp_information": {
            "time_base": time_base_raw if time_base_raw is not None else "not_available",
            "time_base_seconds": clean_float(time_base_seconds, digits=10),
            "start_pts": start_pts,
            "start_time_seconds": clean_float(start_time_seconds),
            "duration_ts": duration_ts if duration_ts is not None else "not_available",
        },
        "gps_or_camera_identifier_metadata": gps_camera_metadata,
        "required_fields_check": {
            "codec_parameters_available": codec_name is not None,
            "frame_rate_available": fps is not None,
            "resolution_available": width is not None and height is not None,
            "duration_available": duration_seconds is not None,
            "frame_count_available": frame_count is not None,
            "timestamp_information_available": (
                time_base_seconds is not None and duration_ts is not None
            ),
            "gps_or_camera_identifier_checked": True,
            "input_handled_gracefully": True,
        },
    }

    return record
