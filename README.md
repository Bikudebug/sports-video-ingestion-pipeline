# Sports Video Ingestion and Preprocessing Pipeline
<img src="./docs/unnamed.png" alt="homework" width="80%">
This repository implements a configurable video ingestion and preprocessing pipeline for sports computer vision workflows. It accepts local video files, folders of videos, and HLS playlist inputs, then converts them into structured, model-ready outputs such as extracted frames, RGB/grayscale images, metadata reports, frame-quality reports, scene-change reports, preview videos, and validation summaries.

The main purpose of this project is to prepare raw sports video data for downstream computer vision tasks. After preprocessing, the exported RGB frames can be used by downstream models such as:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) for multi-person human pose estimation

This repository does **not** train a new detection or pose-estimation model. Instead, it prepares sports video frames so that downstream computer vision models can consume them reliably.
- [For Demostratation the Data and Result](https://drive.google.com/drive/folders/1i54M3LS5oYOmz8g9mu66dQEryvxI-wV1?usp=sharing)
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
        ├── contrast score
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
Downstream computer vision tasks
        ├── YOLO object detection
        └── MMPose multi-person pose estimation
```

The pipeline prepares clean and organized frame outputs. YOLO and MMPose are used only as downstream validation examples to show that the processed frames are suitable for practical computer vision tasks.

---

## 2. Main Features

The pipeline supports:

- Local video files
- Folder-based video ingestion
- Recursive folder scanning
- Local HLS playlist input through `.m3u8`
- Configurable frame extraction
- Configurable target FPS
- Resolution normalization
- BGR to RGB conversion
- Grayscale frame export
- Frame-quality scoring
- Blur, brightness, and contrast analysis
- Poor-quality frame separation
- Scene-change / shot-boundary detection
- Segment-wise frame organization
- Per-frame manifest generation
- Per-video metadata extraction
- Quality report generation
- Scene report generation
- Output validation
- Preview video generation
- Downstream YOLO object detection on processed RGB frames
- Downstream MMPose pose estimation on processed RGB frames

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

Optional downstream tools:

```bash
pip install ultralytics
```

For MMPose, follow the official installation instructions from the OpenMMLab repository:

```text
https://github.com/open-mmlab/mmpose
```

MMPose installation depends on the local CUDA/PyTorch/MMCV environment, so it is kept as an optional downstream dependency rather than a required preprocessing dependency.

---

## 7. Running the Preprocessing Pipeline

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
- contrast threshold
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

Downstream model outputs can be saved separately, for example:

```text
outputs/yolo/
outputs/mmpose/
```

These downstream folders are generated after running object detection or pose estimation.

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

This separation makes it easier to run downstream inference only on reliable frames.

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

This stores frame-quality statistics such as blur, brightness, contrast, and poor-frame counts.

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

The implemented baseline method is:

```yaml
scene_detection:
  method: "hsv_histogram"
```

The HSV histogram method compares consecutive frames using color-distribution differences. Large differences indicate possible scene or shot boundaries. This is useful for sports footage because a video may contain multiple camera views, cuts, replays, close-ups, crowd shots, or broadcast transitions.

Segment-level organization allows downstream computer vision models to process temporally coherent frame groups.

---

## 13. Frame-Quality Analysis

The pipeline computes frame-quality heuristics and separates usable frames from poor-quality frames.

Typical quality checks include:

- blur detection
- brightness analysis
- contrast analysis
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

This helps reduce downstream inference on frames that may be visually unreliable.

---

## 14. Downstream Computer Vision Tasks

The preprocessing pipeline is model-agnostic. Its output can be used by multiple downstream computer vision models.

This project demonstrates two downstream tasks:

```text
Preprocessed RGB frames
        ↓
Downstream CV task
        ├── YOLOv8n object detection
        └── MMPose multi-person pose estimation
```

---

## 15. YOLOv8n Object Detection

The first downstream task is object detection using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

The role of YOLO in this repository is:

```text
Preprocessed RGB frames from this pipeline
        ↓
YOLOv8n object detection
        ↓
Annotated prediction images and detection label files
```

The demonstration model is:

```text
yolov8n.pt
```

This is the lightweight YOLOv8 nano model from Ultralytics.

Install YOLO:

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

YOLO label files follow this format:

```text
class_id x_center y_center width height confidence
```

For COCO-pretrained YOLO models, common useful labels include:

```text
0  = person
32 = sports ball
```

This makes YOLO useful for verifying whether the preprocessed sports frames are suitable for detecting players and balls.

---

## 16. MMPose Multi-Person Pose Estimation

The second downstream task is multi-person human pose estimation using [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose).

The role of MMPose in this repository is:

```text
Preprocessed RGB frames from this pipeline
        ↓
Person detection + pose estimation
        ↓
Player bounding boxes, keypoints, and skeleton overlays
```

Pose estimation is useful because object detection only localizes players with bounding boxes, while pose estimation provides body-joint locations. This is important for sports analysis tasks such as:

- player movement analysis
- running posture analysis
- action phase understanding
- biomechanical motion analysis
- skeleton-based temporal analysis
- player interaction analysis

MMPose can be applied to the exported RGB frame folders, for example:

```text
outputs/playlist/frames/good/segment_000/rgb/
```

A typical MMPose output may include:

```text
outputs/mmpose/
├── visualizations/
├── predictions/
└── pose_results.json
```

The exact command depends on the local MMPose installation, selected detector, selected pose-estimation model, and available device. Follow the official MMPose repository for installation and demo usage:

```text
https://github.com/open-mmlab/mmpose
```

The important point is that the preprocessing pipeline produces normalized and quality-filtered RGB frames that are directly usable for MMPose-based pose estimation.

---

## 17. Example Complete Workflow

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

### Step 6A: Run YOLO object detection

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

YOLO outputs are saved in:

```text
outputs/yolo/segment_000_detection/
```

---

### Step 6B: Run MMPose pose estimation

Use the same RGB frame segment as input to MMPose:

```text
outputs/<video_id>/frames/good/segment_000/rgb
```

The exact command depends on the selected MMPose model and detector configuration. MMPose outputs can be saved under:

```text
outputs/mmpose/
```

---

### Step 7: Review downstream CV results

Possible downstream outputs include:

```text
outputs/yolo/segment_000_detection/
outputs/mmpose/
```

This confirms that the preprocessing pipeline produces model-ready frame data for multiple downstream computer vision tasks.

---

## 18. Git and Large File Policy

Generated outputs should not be committed to Git.

The following types of files should remain local:

- extracted frames
- processed videos
- preview videos
- HLS `.ts` segments
- YOLO prediction outputs
- MMPose prediction outputs
- pose-estimation visualization videos
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

## 19. Summary

This repository provides a complete preprocessing stage for sports computer vision pipelines. It converts raw video or HLS input into structured, validated, frame-level outputs. These outputs are suitable for downstream computer vision tasks such as YOLO-based object detection and MMPose-based human pose estimation.

The main contribution is the video ingestion and preprocessing workflow:

```text
raw sports video
        ↓
validated and normalized frames
        ↓
quality-filtered frame outputs
        ↓
scene-aware frame organization
        ↓
model-ready RGB data
        ↓
downstream computer vision tasks
        ├── object detection
        └── pose estimation
```
