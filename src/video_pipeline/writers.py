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
from .transforms import *


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def draw_label(image_bgr: np.ndarray, text: str, x: int = 10, y: int = 30) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    cv2.putText(
        image_bgr,
        text,
        (x, y),
        font,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def fmt_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "None"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


class OutputVideoWriters:
    def __init__(
        self,
        processed_video_dir: Path,
        preview_dir: Path,
        width: int,
        height: int,
        processed_fps: float,
        preview_fps: float,
        processed_cfg: Dict[str, Any],
        preview_cfg: Dict[str, Any],
    ) -> None:
        self.width = width
        self.height = height
        self.processed_cfg = processed_cfg
        self.preview_cfg = preview_cfg

        self.processed_video_dir = processed_video_dir
        self.preview_dir = preview_dir

        self.processed_video_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.processed_writers: Dict[str, cv2.VideoWriter] = {}
        self.preview_writers: Dict[str, cv2.VideoWriter] = {}
        self.output_paths: Dict[str, str] = {}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        processed_label = fps_label(processed_fps)

        def create_writer(
            path: Path,
            fps: float,
            out_width: int,
            out_height: int,
        ) -> cv2.VideoWriter:
            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                fps,
                (out_width, out_height),
            )

            if not writer.isOpened():
                raise RuntimeError(f"Could not create video writer: {path}")

            return writer

        if bool(processed_cfg.get("save_processed_videos", True)):
            if bool(processed_cfg.get("save_sampled_resized_video", True)):
                path = processed_video_dir / f"sampled_resized_{processed_label}.mp4"
                self.processed_writers["sampled_resized_video"] = create_writer(
                    path,
                    processed_fps,
                    width,
                    height,
                )
                self.output_paths["sampled_resized_video"] = str(path)

            if bool(processed_cfg.get("save_grayscale_video", True)):
                path = processed_video_dir / f"grayscale_{processed_label}.mp4"
                self.processed_writers["grayscale_video"] = create_writer(
                    path,
                    processed_fps,
                    width,
                    height,
                )
                self.output_paths["grayscale_video"] = str(path)

            if bool(processed_cfg.get("save_good_frames_only_video", True)):
                path = processed_video_dir / f"good_frames_only_{processed_label}.mp4"
                self.processed_writers["good_frames_only_video"] = create_writer(
                    path,
                    processed_fps,
                    width,
                    height,
                )
                self.output_paths["good_frames_only_video"] = str(path)

        if bool(preview_cfg.get("save_preview_videos", True)):
            if bool(preview_cfg.get("save_resized_sampled_preview", True)):
                path = preview_dir / "resized_sampled_preview.mp4"
                self.preview_writers["resized_sampled_preview"] = create_writer(
                    path,
                    preview_fps,
                    width,
                    height,
                )
                self.output_paths["resized_sampled_preview"] = str(path)

            if bool(preview_cfg.get("save_grayscale_preview", True)):
                path = preview_dir / "grayscale_preview.mp4"
                self.preview_writers["grayscale_preview"] = create_writer(
                    path,
                    preview_fps,
                    width,
                    height,
                )
                self.output_paths["grayscale_preview"] = str(path)

            if bool(preview_cfg.get("save_side_by_side_transform_preview", True)):
                path = preview_dir / "side_by_side_transform_preview.mp4"
                self.preview_writers["side_by_side_transform_preview"] = create_writer(
                    path,
                    preview_fps,
                    width * 3,
                    height,
                )
                self.output_paths["side_by_side_transform_preview"] = str(path)

            if bool(preview_cfg.get("save_quality_overlay_preview", True)):
                path = preview_dir / "quality_overlay_preview.mp4"
                self.preview_writers["quality_overlay_preview"] = create_writer(
                    path,
                    preview_fps,
                    width,
                    height,
                )
                self.output_paths["quality_overlay_preview"] = str(path)

    def write(
        self,
        resized_bgr: np.ndarray,
        rgb_frame: np.ndarray,
        gray_frame: np.ndarray,
        record: Dict[str, Any],
        preview_count: int,
    ) -> None:
        is_good = bool(record["quality"]["is_good_frame"])

        if "sampled_resized_video" in self.processed_writers:
            self.processed_writers["sampled_resized_video"].write(resized_bgr)

        if "grayscale_video" in self.processed_writers:
            self.processed_writers["grayscale_video"].write(gray_to_bgr(gray_frame))

        if "good_frames_only_video" in self.processed_writers and is_good:
            self.processed_writers["good_frames_only_video"].write(resized_bgr)

        max_preview_frames = int(self.preview_cfg.get("max_preview_frames", 0))

        if max_preview_frames > 0 and preview_count > max_preview_frames:
            return

        frame_index = int(record["frame_index"])
        timestamp_sec = float(record["timestamp_sec"])
        segment_id = int(record["segment_id"])
        status = "GOOD" if is_good else "POOR"

        if "resized_sampled_preview" in self.preview_writers:
            frame = resized_bgr.copy()
            draw_label(
                frame,
                f"Resized sampled | frame={frame_index} | t={timestamp_sec:.3f}s | seg={segment_id}",
                y=28,
            )
            self.preview_writers["resized_sampled_preview"].write(frame)

        if "grayscale_preview" in self.preview_writers:
            frame = gray_to_bgr(gray_frame)
            draw_label(
                frame,
                f"Grayscale | frame={frame_index} | t={timestamp_sec:.3f}s | {status}",
                y=28,
            )
            self.preview_writers["grayscale_preview"].write(frame)

        if "side_by_side_transform_preview" in self.preview_writers:
            left = resized_bgr.copy()
            middle = rgb_to_bgr(rgb_frame)
            right = gray_to_bgr(gray_frame)

            draw_label(left, "Left: resized", y=28)
            draw_label(middle, "Middle: RGB processed", y=28)
            draw_label(right, "Right: grayscale", y=28)

            side_by_side = np.concatenate([left, middle, right], axis=1)

            draw_label(
                side_by_side,
                f"frame={frame_index} | t={timestamp_sec:.3f}s | segment={segment_id} | {status}",
                y=self.height - 18,
            )

            self.preview_writers["side_by_side_transform_preview"].write(side_by_side)

        if "quality_overlay_preview" in self.preview_writers:
            overlay = resized_bgr.copy()
            q = record["quality"]
            sd = record["scene_detection"]

            draw_label(
                overlay,
                f"frame={frame_index} | t={timestamp_sec:.3f}s | segment={segment_id} | {status}",
                y=28,
            )

            draw_label(
                overlay,
                f"blur={q['blur_score']:.2f} | brightness={q['brightness_score']:.2f} | contrast={q['contrast_score']:.2f}",
                y=58,
            )

            draw_label(
                overlay,
                f"scene_diff={fmt_value(sd['scene_difference'])} | scene_change={sd['scene_change']}",
                y=88,
            )

            self.preview_writers["quality_overlay_preview"].write(overlay)

    def release(self) -> None:
        for writer in self.processed_writers.values():
            writer.release()

        for writer in self.preview_writers.values():
            writer.release()


def save_scene_histogram_csv(frame_records: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sampled_frame_number",
        "frame_index",
        "timestamp_sec",
        "segment_id",
        "scene_difference",
        "scene_similarity",
        "scene_change",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in frame_records:
            sd = r["scene_detection"]

            writer.writerow({
                "sampled_frame_number": r["sampled_frame_number"],
                "frame_index": r["frame_index"],
                "timestamp_sec": r["timestamp_sec"],
                "segment_id": r["segment_id"],
                "scene_difference": sd["scene_difference"],
                "scene_similarity": sd["scene_similarity"],
                "scene_change": sd["scene_change"],
            })


def plot_scene_histogram_change(
    frame_records: List[Dict[str, Any]],
    scene_report: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = []
    differences = []
    change_timestamps = []
    change_differences = []

    for r in frame_records:
        sd = r["scene_detection"]
        diff = sd["scene_difference"]

        if diff is None:
            continue

        timestamps.append(float(r["timestamp_sec"]))
        differences.append(float(diff))

        if sd["scene_change"]:
            change_timestamps.append(float(r["timestamp_sec"]))
            change_differences.append(float(diff))

    plt.figure(figsize=(14, 6))

    if len(timestamps) > 0:
        plt.plot(timestamps, differences, linewidth=1.5, label="HSV histogram difference")

    fixed_threshold = scene_report.get("fixed_threshold_used_for_segmentation")
    adaptive_threshold = scene_report.get("adaptive_threshold_suggestion")

    if fixed_threshold is not None:
        plt.axhline(
            y=float(fixed_threshold),
            linestyle="--",
            linewidth=1.2,
            label=f"Fixed threshold = {float(fixed_threshold):.4f}",
        )

    if adaptive_threshold is not None:
        plt.axhline(
            y=float(adaptive_threshold),
            linestyle=":",
            linewidth=1.2,
            label=f"Adaptive p{scene_report.get('adaptive_percentile_for_analysis')} = {float(adaptive_threshold):.4f}",
        )

    if len(change_timestamps) > 0:
        plt.scatter(
            change_timestamps,
            change_differences,
            marker="o",
            label="Detected scene changes",
        )

    plt.title("Scene / Shot-Boundary Analysis using HSV Histogram Difference")
    plt.xlabel("Timestamp (seconds)")
    plt.ylabel("Scene difference = 1 - histogram correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_quality_scores(frame_records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = [float(r["timestamp_sec"]) for r in frame_records]
    blur_scores = [float(r["quality"]["blur_score"]) for r in frame_records]
    brightness_scores = [float(r["quality"]["brightness_score"]) for r in frame_records]
    contrast_scores = [float(r["quality"]["contrast_score"]) for r in frame_records]

    plt.figure(figsize=(14, 6))

    if len(timestamps) > 0:
        plt.plot(timestamps, blur_scores, label="Blur score")
        plt.plot(timestamps, brightness_scores, label="Brightness score")
        plt.plot(timestamps, contrast_scores, label="Contrast score")

    plt.title("Frame Quality Scores over Time")
    plt.xlabel("Timestamp (seconds)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
