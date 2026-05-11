# Sports Video Ingestion and Preprocessing Pipeline

This repository implements a configurable video ingestion and preprocessing pipeline for sports computer vision workflows. It accepts local video files, folders of videos, and HLS playlist inputs, then converts them into structured, model-ready outputs such as extracted frames, RGB/grayscale images, metadata reports, frame-quality reports, scene-change reports, preview videos, and validation summaries.

The main purpose of this project is to prepare raw sports video data for downstream computer vision tasks. After preprocessing, the exported RGB frames can be used directly by object detection models such as YOLO.

---

## 1. End-to-End Workflow

The complete workflow is:

```text
Input sports video / folder / HLS playlist
        ↓
Input validation and metadata extraction
        ↓
Video decoding
        ↓
Configurable frame sampling
        ↓
Resolution normalization
        ↓
Color-space conversion
        ├── RGB frame export
        └── grayscale frame export
        ↓
Frame-quality analysis
        ├── blur score
        ├── brightness score
        └── poor-frame flagging
        ↓
Scene-change / shot-boundary detection
        ↓
Structured frame export by detected segment
        ↓
Manifest and report generation
        ├── frame manifest
        ├── quality report
        ├── scene report
        └── validation report
        ↓
Downstream computer vision inference
        └── YOLO object detection on exported RGB frames
```

This repository does **not** train an object detection model. Instead, it prepares sports video data so that a downstream model can consume the processed frames reliably.

---

## 2. Main Features

The pipeline supports:

- Local video files
- Folder-based video ingestion
- Recursive folder scanning
- Local HLS playlist input through `.m3u8`
- Configurable frame extraction
- Resolution normalization
- BGR to RGB conversion
- Grayscale frame export
- Frame-quality scoring
- Poor-quality frame separation
- Scene-change / shot-boundary detection
- Per-frame manifest generation
- Per-video metadata extraction
- Quality report generation
- Scene report generation
- Output validation
- Preview video generation
- Downstream YOLO object detection on processed RGB frames

---

## 3. Supported Input Formats

The pipeline is designed to support common video and stream-style inputs such as:

```text
.mp4
.mkv
.avi
.mov
.webm
.m4v
.mts
.m2ts
.mpeg
.mpg
.m3u8
```

For HLS input, the `.m3u8` playlist can be passed directly as the input path.

---

## 4. Project Structure

```text
.
├── configs/
│   └── video_processing_analysis.yaml
├── dataset/
├── docs/
├── outputs/
├── src/
│   ├── video_processing_analysis_pipeline.py
│   └── video_pipeline/
│       ├── cli.py
│       ├── common.py
│       ├── config.py
│       ├── inputs.py
│       ├── metadata.py
│       ├── pipeline.py
│       ├── quality.py
│       ├── scenes.py
│       ├── transforms.py
│       ├── validation.py
│       └── writers.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 5. Main Entry Point

The main executable script is:

```text
src/video_processing_analysis_pipeline.py
```

Internally, this script calls:

```text
video_pipeline.cli.main()
```

The main pipeline orchestration is implemented in:

```text
src/video_pipeline/pipeline.py
```

---

## 6. Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg and FFprobe must also be installed and available on the system path. They are required for video decoding, metadata extraction, processed-video generation, preview generation, and output validation.

Check FFmpeg and FFprobe:

```bash
ffmpeg -version
ffprobe -version
```

---

## 7. Running the Pipeline

### 7.1 Run with the default configuration

```bash
python src/video_processing_analysis_pipeline.py --config configs/video_processing_analysis.yaml
```

---

### 7.2 Run with a specific video file

```bash
python src/video_processing_analysis_pipeline.py \
  --config configs/video_processing_analysis.yaml \
  --input dataset/sample.mp4 \
  --output outputs
```

---

### 7.3 Run with a local HLS playlist

```bash
python src/video_processing_analysis_pipeline.py \
  --config configs/video_processing_analysis.yaml \
  --input dataset/hls/playlist.m3u8 \
  --output outputs
```

On Windows PowerShell:

```powershell
python src/video_processing_analysis_pipeline.py `
  --config configs/video_processing_analysis.yaml `
  --input "C:\path\to\playlist.m3u8" `
  --output outputs
```

---

### 7.4 Run on a folder of videos

```bash
python src/video_processing_analysis_pipeline.py \
  --config configs/video_processing_analysis.yaml \
  --input dataset \
  --output outputs
```

---

### 7.5 Run recursively on videos inside subfolders

```bash
python src/video_processing_analysis_pipeline.py \
  --config configs/video_processing_analysis.yaml \
  --input dataset \
  --output outputs \
  --recursive
```

---

## 8. Configuration

The main configuration file is:

```text
configs/video_processing_analysis.yaml
```

It controls the major preprocessing parameters, including:

- input source
- output directory
- supported input extensions
- frame sampling mode
- target FPS
- target resize resolution
- RGB frame export
- grayscale frame export
- frame-quality thresholds
- blur detection threshold
- brightness threshold
- scene-change detection threshold
- preview generation
- processed-video generation
- validation settings
- logging level

This makes the pipeline configurable without changing the source code.

---

## 9. Output Structure

For each input video or HLS stream, the pipeline creates a separate output directory. For example, if the input is named `playlist.m3u8`, the output may look like:

```text
outputs/playlist/
```

A typical output structure is:

```text
outputs/playlist/
├── embedded_metadata/
│   └── metadata.json
├── frames/
│   ├── good/
│   │   ├── segment_000/
│   │   │   ├── rgb/
│   │   │   └── gray/
│   │   ├── segment_001/
│   │   │   ├── rgb/
│   │   │   └── gray/
│   │   └── ...
│   └── poor_quality/
├── manifests/
│   ├── manifest.json
│   ├── frames_manifest.jsonl
│   ├── quality_report.json
│   ├── scene_report.json
│   └── validation_report.json
├── processed_videos/
├── previews/
└── analysis/
```

---

## 10. Frame Export

Good-quality frames are grouped by detected scene segment:

```text
outputs/playlist/frames/good/segment_000/rgb/
outputs/playlist/frames/good/segment_000/gray/
outputs/playlist/frames/good/segment_001/rgb/
outputs/playlist/frames/good/segment_001/gray/
```

Poor-quality frames are separated into:

```text
outputs/playlist/frames/poor_quality/
```

This separation makes it easier to use only reliable frames for downstream model inference.

---

## 11. Manifest Files

The pipeline generates structured manifest and report files.

### Main manifest

```text
outputs/playlist/manifests/manifest.json
```

This stores high-level processing information for the input video.

### Frame manifest

```text
outputs/playlist/manifests/frames_manifest.jsonl
```

This file stores one record per exported frame. Each row can include information such as:

- frame index
- timestamp
- output path
- RGB frame path
- grayscale frame path
- scene/segment ID
- quality score
- poor-quality flag

This file is useful when connecting the preprocessing output to another computer vision service.

### Quality report

```text
outputs/playlist/manifests/quality_report.json
```

This stores frame-quality statistics such as blur, brightness, and poor-frame counts.

### Scene report

```text
outputs/playlist/manifests/scene_report.json
```

This stores detected scene-change or shot-boundary information.

### Validation report

```text
outputs/playlist/manifests/validation_report.json
```

This confirms whether the output structure and generated files are valid.

---

## 12. Scene-Change Detection

The pipeline detects scene changes or shot boundaries and groups good frames according to segment IDs.

Example:

```text
segment_000/
segment_001/
segment_002/
```

This is useful for sports footage because a video may contain multiple camera views, cuts, replays, or transitions. Segment-level organization allows downstream computer vision models to process temporally coherent frame groups.

---

## 13. Frame-Quality Analysis

The pipeline computes frame-quality heuristics and separates usable frames from poor-quality frames.

Typical quality checks include:

- blur detection
- brightness analysis
- frame validity checks
- poor-frame flagging

Good frames are saved under:

```text
frames/good/
```

Poor-quality frames are saved under:

```text
frames/poor_quality/
```

---

## 14. Downstream Computer Vision Integration with YOLO

The assessment requires that the processed output should be usable by a downstream computer vision model. This project demonstrates that requirement using YOLO object detection.

The role of YOLO in this repository is:

```text
Preprocessed RGB frames from this pipeline
        ↓
YOLO object detection
        ↓
Annotated prediction images and detection label files
```

The pipeline first prepares clean frame data. YOLO is then applied to the exported RGB frames to verify that the output can be consumed by a standard object detection model.

The demonstration model is:

```text
yolov8n.pt
```
This is the lightweight YOLOv8 nano model from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

Install YOLO separately:

```bash
pip install ultralytics
```

Run YOLO on one exported RGB segment:

```bash
yolo predict \
  model=yolov8n.pt \
  source="outputs/playlist/frames/good/segment_000/rgb" \
  device=cpu \
  conf=0.25 \
  save=True \
  save_txt=True \
  save_conf=True \
  project="outputs/yolo" \
  name="segment_000_detection"
```

If a CUDA GPU is available, use:

```bash
device=0
```

instead of:

```bash
device=cpu
```

The YOLO output will be saved under:

```text
outputs/yolo/segment_000_detection/
```

This folder contains annotated prediction images and detection label files.

---

## 15. Example Complete Workflow

A complete run follows these steps.

### Step 1: Place input data locally

Place a video file or HLS playlist under:

```text
dataset/
```

Example:

```text
dataset/sample.mp4
```

or:

```text
dataset/hls/playlist.m3u8
```

---

### Step 2: Run the preprocessing pipeline

```bash
python src/video_processing_analysis_pipeline.py \
  --config configs/video_processing_analysis.yaml \
  --input dataset/sample.mp4 \
  --output outputs
```

---

### Step 3: Check validation output

Open:

```text
outputs/<video_id>/manifests/validation_report.json
```

Confirm that the output validation has passed.

---

### Step 4: Inspect scene segmentation

Open:

```text
outputs/<video_id>/manifests/scene_report.json
```

This file describes detected scene changes and segment information.

---

### Step 5: Select a good RGB frame segment

Example:

```text
outputs/<video_id>/frames/good/segment_000/rgb
```

---

### Step 6: Run YOLO object detection

```bash
yolo predict \
  model=yolov8n.pt \
  source="outputs/<video_id>/frames/good/segment_000/rgb" \
  device=cpu \
  conf=0.25 \
  save=True \
  save_txt=True \
  save_conf=True \
  project="outputs/yolo" \
  name="segment_000_detection"
```

---

### Step 7: Review YOLO results

YOLO outputs are saved in:

```text
outputs/yolo/segment_000_detection/
```

This confirms that the preprocessing pipeline produces model-ready frame data for downstream computer vision inference.

---

## 16. Git and Large File Policy

Generated outputs should not be committed to Git.

The following types of files should remain local:

- extracted frames
- processed videos
- preview videos
- HLS `.ts` segments
- YOLO prediction outputs
- large input videos
- temporary analysis files

The repository should track source code, configuration files, tests, documentation, and lightweight placeholder files such as `.gitkeep`.

Input videos can be kept locally under:

```text
dataset/
```

Output files can be generated locally under:

```text
outputs/
```

---

## 17. Summary

This repository provides a complete preprocessing stage for sports computer vision pipelines. It converts raw video or HLS input into structured, validated, frame-level outputs. These outputs are then suitable for downstream computer vision tasks such as YOLO-based object detection.

The main contribution is the video ingestion and preprocessing workflow:

```text
raw sports video
        ↓
validated and normalized frames
        ↓
scene-aware frame organization
        ↓
quality-filtered model-ready data
        ↓
downstream object detection
```
