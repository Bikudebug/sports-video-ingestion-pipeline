#!/usr/bin/env python3
"""
Command-line entry point for the video processing analysis pipeline.

Recommended run:
    python src/video_processing_analysis_pipeline.py --config configs/video_processing_analysis.yaml

The YAML file stores the full pipeline configuration. You can still override
the input video/folder/stream and output folder directly from the command line:
    python src/video_processing_analysis_pipeline.py --config configs/video_processing_analysis.yaml --input dataset/sample.mp4 --output outputs

This main file is intentionally small. The actual pipeline functions are split
inside src/video_pipeline/ and are called through video_pipeline.cli.main().

If you want a future version to run without a YAML file, add CLI arguments in
video_pipeline/cli.py for every required config value, then build the config
dictionary from those arguments before calling the pipeline.
"""

from video_pipeline.cli import main


if __name__ == "__main__":
    main()
