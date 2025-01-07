# Video Compression and Playback

This project contains Python scripts for performing video compression and playback using Discrete Cosine Transform (DCT), motion vector analysis, and binary encoding of video frames. It includes scripts for encoding a video file, compressing it, and playing it back in synchronization with audio.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Compressing the Video](#step-1-compressing-the-video)
  - [Step 2: Playing the Compressed Video](#step-2-playing-the-compressed-video)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Exploration of Detectron Technology](#exploration-of-detectron-technology)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a simplified video compression and playback system:

1. Video compression is performed by dividing video frames into blocks, applying DCT, quantization, and encoding motion vectors.
2. Playback of the compressed video is performed by decoding the binary file and synchronizing video with audio.

## Features

- **Video Compression**: Compresses raw `.rgb` video files using DCT, motion vector computation, and quantization.
- **Binary Encoding**: Encodes compressed frames and macroblocks into a `.cmp` binary file.
- **Playback**: Decodes the `.cmp` file and synchronizes it with an accompanying `.wav` audio file.
- **Visualization**: Displays segmented regions (foreground and background) during compression for analysis.

## Installation

### Test Data

To test the pipeline, you can use the RGB and WAV files available [here](https://drive.google.com/drive/folders/1Zobg8iIJhoISsk13NJztsQJ0zCFL-J5y).

## Usage

### Step 1: Compressing the Video

Run the `Encode.py` script to compress the video:

```bash
python Encode.py
```

**Inputs:**

- `3.rgb`: Raw RGB video file with resolution 960x540.
- `compressed_output.cmp`: Output binary file containing compressed video.
- Quantization parameters for foreground (`n1`) and background (`n2`).

### Step 2: Playing the Compressed Video

Run the `processandplay.py` script to play the compressed video:

```bash
python processandplay.py
```

**Inputs:**

- `compressed_output.cmp`: Compressed binary video file.
- `3.wav`: Accompanying audio file.
- Frame rate (FPS): 30.

**Key Controls During Playback:**

- `q`: Quit playback.
- `p`: Pause playback.
- `s`: Step forward frame-by-frame when paused.

## File Descriptions

### `dct.py`

Contains utility functions for DCT and IDCT transformations, zigzag coefficient selection, and image processing:

- `perform_dct(block)`: Applies DCT to an 8x8 block.
- `perform_idct(block)`: Applies inverse DCT to an 8x8 block.
- `read_image_rgb(file_path, width, height)`: Reads a raw RGB image file.
- `save_image(r, g, b, output_path)`: Saves RGB channels into an image.
- `process_image_dct(r, g, b, n)`: Processes an image using DCT and zigzag coefficient selection.

### `Encode.py`

Encodes raw video into a compressed binary `.cmp` file:

- Computes motion vectors using `compute_motion_vector_tss`.
- Classifies macroblocks as foreground or background.
- Applies DCT, quantization, and writes compressed data into the binary file.

### `processandplay.py`

Plays the compressed `.cmp` video and synchronizes it with audio:

- Decodes binary video data.
- Applies IDCT and reconstructs video frames.
- Synchronizes with audio using PyAudio.

### `Step1_code.py`

Provides helper functions for computing motion vectors and segmenting frames:

- `compute_motion_vector_tss`: Implements Three-Step Search (TSS) for motion estimation.
- `classify_macroblocks_r`: Classifies macroblocks based on motion vector magnitude.
- `visualize_segmentation`: Highlights foreground blocks during compression.

### `processandplay.py`

Synchronizes and plays the compressed video with audio.

- Preprocesses frames from the `.cmp` file.
- Reads and processes the audio for synchronized playback.

## Exploration of Detectron Technology

The `step1_detectron2.py` file explores the use of Detectron2 for object detection and video segmentation. This script integrates advanced machine learning techniques for analyzing and highlighting moving objects in a video.

**Features:**

- **Object Detection:** Uses the Detectron2 framework to detect and segment objects within each frame of the video.
- **Optical Flow Analysis:** Computes dense optical flow between consecutive frames to identify motion vectors for each pixel.
- **Camera Motion Detection:** Differentiates between camera motion and object motion using global motion vectors.
- **Foreground Segmentation:** Filters and highlights moving objects by combining optical flow and object detection results.

**Visualization:**

- Foreground objects are highlighted in red blocks, making it easier to analyze moving objects in the video.

**Usage:**

1. Install the additional dependencies for Detectron2:
   ```bash
   pip install torch torchvision
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
   ```
2. Run the script:
   ```bash
   python step1_detectron2.py
   ```

**Inputs:**

- `1.rgb`: Raw RGB video file.
- Frame resolution: 960x540.

This script serves as a demonstration of Detectron2's capabilities and is independent of the main encoding-decoding pipeline.

## Dependencies

- **Python 3.6+**
- Libraries:
  - `numpy`
  - `opencv-python`
  - `scipy`
  - `pyaudio`
  - `Pillow`
  - `torch` and `torchvision` (for Detectron2)
  - `detectron2`

