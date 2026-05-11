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
from .transforms import *
from .quality import *
from .scenes import *
from .writers import *
from .validation import *


def preprocess_source(
    input_source: str,
    video_id: str,
    video_output_dir: Path,
    metadata_path: Path,
    source_metadata: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    frames_root = video_output_dir / "frames"
    processed_video_dir = video_output_dir / "processed_videos"
    preview_dir = video_output_dir / "previews"
    analysis_dir = video_output_dir / "analysis"
    manifest_dir = video_output_dir / "manifests"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    cap = open_video(input_source)

    sampling_cfg = config.get("sampling", {})
    transform_cfg = config.get("transform", {})
    quality_cfg = config.get("quality", {})
    scene_cfg = config.get("scene_detection", {})
    processed_cfg = config.get("processed_videos", {})
    preview_cfg = config.get("preview", {})
    analysis_cfg = config.get("analysis", {})
    output_cfg = config.get("output", {})

    source_fps_fallback = float(sampling_cfg.get("source_fps_fallback", 30.0))

    cv_metadata = read_video_metadata_from_capture(
        cap=cap,
        input_source=input_source,
        config=config,
        source_fps_fallback=source_fps_fallback,
    )

    fps_used = float(cv_metadata["fps_used"])

    scene_method = str(scene_cfg.get("method", "hsv_histogram")).lower()

    if scene_method != "hsv_histogram":
        raise NotImplementedError(
            f"Scene detection method '{scene_method}' is not implemented in this script. "
            "Use method: hsv_histogram, or implement the selected method."
        )

    sample_interval = compute_sampling_interval(
        fps_used=fps_used,
        sampling_cfg=sampling_cfg,
    )

    effective_sample_fps = fps_used / sample_interval if sample_interval > 0 else None
    sample_dt_sec = 1.0 / effective_sample_fps if effective_sample_fps and effective_sample_fps > 0 else None

    target_width = int(transform_cfg.get("target_width", 640))
    target_height = int(transform_cfg.get("target_height", 360))

    image_format = str(output_cfg.get("image_format", "jpg")).lower()
    if image_format.startswith("."):
        image_format = image_format[1:]

    save_poor = bool(quality_cfg.get("save_poor_quality_frames", True))
    save_gray = bool(transform_cfg.get("save_grayscale_frames", True))

    processed_fps = float(
        processed_cfg.get(
            "output_fps",
            sampling_cfg.get("target_fps", 5),
        )
    )

    preview_fps = float(
        preview_cfg.get(
            "preview_fps",
            sampling_cfg.get("target_fps", 5),
        )
    )

    video_writers = OutputVideoWriters(
        processed_video_dir=processed_video_dir,
        preview_dir=preview_dir,
        width=target_width,
        height=target_height,
        processed_fps=processed_fps,
        preview_fps=preview_fps,
        processed_cfg=processed_cfg,
        preview_cfg=preview_cfg,
    )

    logger.info("Starting video preprocessing and analysis.")
    logger.info(f"Source: {input_source}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Sampling interval: every {sample_interval} frame(s)")
    logger.info(f"Target resolution: {target_width}x{target_height}")

    frame_records: List[Dict[str, Any]] = []

    total_frames_read = 0
    sampled_frames = 0
    saved_good_rgb_frames = 0
    saved_good_gray_frames = 0
    saved_poor_rgb_frames = 0
    saved_poor_gray_frames = 0
    flagged_poor_frames = 0

    scene_threshold = float(scene_cfg.get("threshold", 0.45))
    min_scene_gap = int(scene_cfg.get("min_scene_gap_sampled_frames", 10))

    current_segment_id = 0
    last_scene_change_sampled_frame = -10**9
    previous_hist: Optional[np.ndarray] = None
    preview_count = 0
    start_time = time.perf_counter()
    frame_index = -1

    while True:
        success, frame_bgr = cap.read()

        if not success:
            break

        frame_index += 1
        total_frames_read += 1

        if frame_index % sample_interval != 0:
            continue

        sampled_frames += 1

        timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))

        if timestamp_ms > 0:
            timestamp_sec = timestamp_ms / 1000.0
        else:
            timestamp_sec = frame_index / fps_used if fps_used > 0 else 0.0

        resized_bgr = resize_frame_bgr(
            frame_bgr=frame_bgr,
            target_width=target_width,
            target_height=target_height,
        )

        rgb_frame = bgr_to_rgb(resized_bgr)
        gray_frame = bgr_to_gray(resized_bgr)

        quality = compute_quality_scores(
            frame_gray=gray_frame,
            quality_cfg=quality_cfg,
        )

        is_good_frame = bool(quality["is_good_frame"])

        if not is_good_frame:
            flagged_poor_frames += 1

        current_hist = compute_hsv_histogram(resized_bgr)

        scene_difference, scene_similarity = compare_histograms(
            previous_hist=previous_hist,
            current_hist=current_hist,
        )

        previous_hist = current_hist

        current_segment_id, scene_change, last_scene_change_sampled_frame = update_segment_id(
            current_segment_id=current_segment_id,
            sampled_frame_number=sampled_frames,
            scene_difference=scene_difference,
            scene_threshold=scene_threshold,
            min_scene_gap=min_scene_gap,
            last_scene_change_sampled_frame=last_scene_change_sampled_frame,
        )

        frame_name = f"frame_{frame_index:08d}_t{timestamp_sec:010.3f}s.{image_format}"
        quality_folder = "good" if is_good_frame else "poor_quality"

        rgb_output_path = None
        gray_output_path = None

        if is_good_frame or save_poor:
            rgb_path = (
                frames_root /
                quality_folder /
                f"segment_{current_segment_id:03d}" /
                "rgb" /
                frame_name
            )

            save_image(rgb_path, rgb_to_bgr(rgb_frame))
            rgb_output_path = str(rgb_path)

            if save_gray:
                gray_path = (
                    frames_root /
                    quality_folder /
                    f"segment_{current_segment_id:03d}" /
                    "gray" /
                    frame_name
                )

                save_image(gray_path, gray_frame)
                gray_output_path = str(gray_path)

            if is_good_frame:
                saved_good_rgb_frames += 1
                if save_gray:
                    saved_good_gray_frames += 1
            else:
                saved_poor_rgb_frames += 1
                if save_gray:
                    saved_poor_gray_frames += 1

        relative_rgb_path = make_relative_path(rgb_output_path, video_output_dir) if rgb_output_path else None
        relative_gray_path = make_relative_path(gray_output_path, video_output_dir) if gray_output_path else None

        record = {
            "source_video_id": video_id,
            "source_video_path": input_source,
            "frame_index": int(frame_index),
            "sampled_frame_number": int(sampled_frames),
            "timestamp_sec": float(timestamp_sec),
            "segment_id": int(current_segment_id),
            "saved": rgb_output_path is not None,
            "output_paths": {
                "rgb": rgb_output_path,
                "gray": gray_output_path,
            },
            "relative_paths": {
                "rgb": relative_rgb_path,
                "gray": relative_gray_path,
            },
            "inference_input": {
                "image_path": rgb_output_path,
                "relative_image_path": relative_rgb_path,
                "image_type": "rgb",
                "valid_for_inference": bool(rgb_output_path is not None and is_good_frame),
            },
            "processed_shape": {
                "height": target_height,
                "width": target_width,
                "channels_rgb": 3,
                "channels_gray": 1 if save_gray else 0,
            },
            "temporal_info": {
                "source_fps": fps_used,
                "sample_interval_frames": sample_interval,
                "effective_sample_fps": effective_sample_fps,
                "sample_dt_sec": sample_dt_sec,
            },
            "transformations": {
                "resize": {
                    "from_width": cv_metadata["width"],
                    "from_height": cv_metadata["height"],
                    "target_width": target_width,
                    "target_height": target_height,
                },
                "color_space": {
                    "opencv_decoded_as": "BGR",
                    "rgb_output": "BGR_to_RGB",
                    "grayscale_output": "BGR_to_GRAYSCALE" if save_gray else "not_saved",
                },
            },
            "quality": quality,
            "scene_detection": {
                "method": scene_method,
                "applied_on": "resized_sampled_frame",
                "scene_difference": scene_difference,
                "scene_similarity": scene_similarity,
                "scene_change": bool(scene_change),
                "threshold_used": scene_threshold,
            },
        }

        frame_records.append(record)
        preview_count += 1

        video_writers.write(
            resized_bgr=resized_bgr,
            rgb_frame=rgb_frame,
            gray_frame=gray_frame,
            record=record,
            preview_count=preview_count,
        )

        if sampled_frames % 100 == 0:
            logger.info(
                f"Sampled frames: {sampled_frames} | "
                f"Good RGB saved: {saved_good_rgb_frames} | "
                f"Poor flagged: {flagged_poor_frames} | "
                f"Current segment: {current_segment_id}"
            )

    cap.release()
    video_writers.release()

    elapsed_sec = time.perf_counter() - start_time

    throughput_read_fps = (
        total_frames_read / elapsed_sec
        if elapsed_sec > 0 else None
    )

    throughput_sampled_fps = (
        sampled_frames / elapsed_sec
        if elapsed_sec > 0 else None
    )

    quality_report = create_quality_report(
        frame_records=frame_records,
        quality_cfg=quality_cfg,
    )

    scene_report = create_scene_report(
        frame_records=frame_records,
        scene_cfg=scene_cfg,
    )

    scene_csv_path = analysis_dir / "scene_histogram_change.csv"
    scene_plot_path = analysis_dir / "scene_histogram_change.png"
    quality_plot_path = analysis_dir / "quality_scores_plot.png"

    analysis_paths: Dict[str, str] = {}

    if bool(analysis_cfg.get("save_scene_histogram_csv", True)):
        save_scene_histogram_csv(
            frame_records=frame_records,
            csv_path=scene_csv_path,
        )
        analysis_paths["scene_histogram_change_csv"] = str(scene_csv_path)

    if bool(analysis_cfg.get("save_scene_histogram_graph", True)):
        plot_scene_histogram_change(
            frame_records=frame_records,
            scene_report=scene_report,
            output_path=scene_plot_path,
        )
        analysis_paths["scene_histogram_change_graph"] = str(scene_plot_path)

    if bool(analysis_cfg.get("save_quality_scores_graph", True)):
        plot_quality_scores(
            frame_records=frame_records,
            output_path=quality_plot_path,
        )
        analysis_paths["quality_scores_plot"] = str(quality_plot_path)

    summary = {
        "report_type": "video_processing_summary",
        "version": "professional_v1",
        "video_id": video_id,
        "input_video": input_source,
        "output_directory": str(video_output_dir),
        "source_metadata_status": source_metadata.get("status"),
        "source_metadata_cv2": cv_metadata,
        "sampling": {
            "mode": sampling_cfg.get("mode"),
            "source_fps_used": fps_used,
            "target_fps": sampling_cfg.get("target_fps"),
            "sample_interval": sample_interval,
            "effective_sample_fps": effective_sample_fps,
            "sample_dt_sec": sample_dt_sec,
            "sampled_frames": sampled_frames,
            "total_frames_read": total_frames_read,
        },
        "target_resolution": {
            "width": target_width,
            "height": target_height,
        },
        "color_outputs": {
            "opencv_decoded_as": "BGR",
            "rgb_saved": True,
            "grayscale_saved": save_gray,
        },
        "saved_frames": {
            "good_rgb_frames": saved_good_rgb_frames,
            "good_gray_frames": saved_good_gray_frames,
            "poor_rgb_frames": saved_poor_rgb_frames,
            "poor_gray_frames": saved_poor_gray_frames,
            "flagged_poor_frames": flagged_poor_frames,
        },
        "scene_detection": {
            "scene_changes_detected": scene_report["scene_changes_detected"],
            "total_segments": scene_report["total_segments"],
            "fixed_threshold_used": scene_report["fixed_threshold_used_for_segmentation"],
            "adaptive_threshold_suggestion": scene_report["adaptive_threshold_suggestion"],
        },
        "processed_video_outputs": {
            "processed_video_directory": str(processed_video_dir),
            **{
                key: value
                for key, value in video_writers.output_paths.items()
                if key in {
                    "sampled_resized_video",
                    "grayscale_video",
                    "good_frames_only_video",
                }
            },
        },
        "preview_outputs": {
            "preview_directory": str(preview_dir),
            **{
                key: value
                for key, value in video_writers.output_paths.items()
                if key in {
                    "resized_sampled_preview",
                    "grayscale_preview",
                    "side_by_side_transform_preview",
                    "quality_overlay_preview",
                }
            },
        },
        "analysis_outputs": {
            "analysis_directory": str(analysis_dir),
            **analysis_paths,
        },
        "performance": {
            "elapsed_sec": elapsed_sec,
            "throughput_read_fps": throughput_read_fps,
            "throughput_sampled_fps": throughput_sampled_fps,
        },
    }

    manifest_json_path = manifest_dir / "manifest.json"
    manifest_jsonl_path = manifest_dir / "frames_manifest.jsonl"
    summary_json_path = manifest_dir / "summary.json"
    quality_report_path = manifest_dir / "quality_report.json"
    scene_report_path = manifest_dir / "scene_report.json"
    validation_report_path = manifest_dir / "validation_report.json"

    manifest_paths = {
        "manifest_json": str(manifest_json_path),
        "frames_manifest_jsonl": str(manifest_jsonl_path),
        "summary_json": str(summary_json_path),
        "quality_report_json": str(quality_report_path),
        "scene_report_json": str(scene_report_path),
    }

    full_manifest = {
        "pipeline": "Professional Video Processing and Analysis Pipeline",
        "created_utc": utc_now_iso(),
        "source_metadata": source_metadata,
        "source_metadata_cv2": cv_metadata,
        "config_used": config,
        "summary": summary,
        "quality_report": quality_report,
        "scene_report": scene_report,
        "frames": frame_records,
    }

    write_json(full_manifest, manifest_json_path)

    with manifest_jsonl_path.open("w", encoding="utf-8") as f:
        for record in frame_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(summary, summary_json_path)
    write_json(quality_report, quality_report_path)
    write_json(scene_report, scene_report_path)

    validation_report = validate_outputs(
        video_output_dir=video_output_dir,
        metadata_path=metadata_path,
        frame_records=frame_records,
        summary=summary,
        manifest_paths=manifest_paths,
        video_output_paths=video_writers.output_paths,
        analysis_paths=analysis_paths,
        config=config,
    )

    write_json(validation_report, validation_report_path)

    manifest_paths["validation_report_json"] = str(validation_report_path)

    full_manifest["validation_report"] = validation_report
    full_manifest["manifest_paths"] = manifest_paths

    write_json(full_manifest, manifest_json_path)

    logger.info("Video preprocessing and analysis completed.")
    logger.info("Output validation completed.")
    logger.info(f"Manifest JSON: {manifest_json_path}")
    logger.info(f"Frames JSONL: {manifest_jsonl_path}")
    logger.info(f"Validation report: {validation_report_path}")
    logger.info(f"Validation status: {validation_report['status']}")
    logger.info(f"Total frames read: {total_frames_read}")
    logger.info(f"Sampled frames: {sampled_frames}")
    logger.info(f"Flagged poor frames: {flagged_poor_frames}")
    logger.info(f"Scene changes detected: {scene_report['scene_changes_detected']}")
    logger.info(f"Total segments: {scene_report['total_segments']}")
    logger.info(f"Elapsed time: {elapsed_sec:.3f} sec")

    return full_manifest


def process_one_source(
    input_source: str,
    config: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    output_cfg = config.get("output", {})
    pipeline_cfg = config.get("pipeline", {})

    append_hash = bool(output_cfg.get("append_source_hash_to_video_id", False))

    video_id = safe_video_id(
        source=input_source,
        config=config,
        append_hash=append_hash,
    )

    video_output_dir = output_root / video_id
    metadata_dir = video_output_dir / "embedded_metadata"
    logs_dir = video_output_dir / "logs"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        log_dir=logs_dir,
        level=config.get("logging", {}).get("level", "INFO"),
    )

    logger.info("============================================================")
    logger.info("Starting video processing and analysis for one input.")
    logger.info(f"Input source: {input_source}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Output directory: {video_output_dir}")

    metadata_path = metadata_dir / "metadata.json"

    if bool(pipeline_cfg.get("run_metadata_extraction", True)):
        try:
            logger.info("Extracting embedded metadata and file-health information.")
            source_metadata = build_source_metadata(
                input_source=input_source,
                config=config,
            )
            write_json(source_metadata, metadata_path)
            logger.info(f"Metadata saved: {metadata_path}")
        except Exception as exc:
            error_type = classify_error(str(exc))
            source_metadata = build_metadata_error_record(
                input_source=input_source,
                config=config,
                error_type=error_type,
                message=str(exc),
            )
            write_json(source_metadata, metadata_path)
            logger.error(f"Metadata extraction failed gracefully. Error JSON saved: {metadata_path}")
            logger.error(short_error(str(exc), limit=800))
    else:
        source_metadata = {
            "status": "skipped",
            "input": {
                "source_path": input_source,
                "file_name": source_name(input_source, config),
                "file_extension": source_extension(input_source, config),
                "is_stream": is_stream_source(input_source, config),
            },
        }
        write_json(source_metadata, metadata_path)

    processing_manifest: Optional[Dict[str, Any]] = None

    should_skip_processing = (
        source_metadata.get("status") == "failed" and
        bool(pipeline_cfg.get("skip_preprocessing_if_metadata_fails", True))
    )

    if should_skip_processing:
        logger.warning(
            "Skipping preprocessing because metadata extraction failed "
            "and skip_preprocessing_if_metadata_fails=true."
        )
    elif bool(pipeline_cfg.get("run_preprocessing", True)):
        processing_manifest = preprocess_source(
            input_source=input_source,
            video_id=video_id,
            video_output_dir=video_output_dir,
            metadata_path=metadata_path,
            source_metadata=source_metadata,
            config=config,
            logger=logger,
        )
    else:
        logger.info("Preprocessing skipped by configuration.")

    final_status = "success"

    if source_metadata.get("status") == "failed":
        final_status = "failed_metadata"

    if (
        processing_manifest is None and
        not should_skip_processing and
        bool(pipeline_cfg.get("run_preprocessing", True))
    ):
        final_status = "failed_preprocessing"

    validation_status = "not_run"
    if processing_manifest is not None:
        validation_status = processing_manifest.get("validation_report", {}).get("status", "not_run")

    return {
        "pipeline": "Professional Video Processing and Analysis Pipeline",
        "created_utc": utc_now_iso(),
        "input_source": input_source,
        "video_id": video_id,
        "output_directory": str(video_output_dir),
        "metadata_path": str(metadata_path),
        "metadata_status": source_metadata.get("status"),
        "preprocessing_ran": processing_manifest is not None,
        "validation_status": validation_status,
        "final_status": final_status,
    }
