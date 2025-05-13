# AVSE-Cognition-AI: Audio-Visual Speech Enhancement with Cognitive Understanding

A real-time system for audio-visual speech enhancement (AVSE) that combines audio, visual, and MEG signals to improve speech quality and understanding. The system uses deep learning to fuse multimodal data and provides real-time transcription and contextual understanding.

## Features

- Real-time processing of audio, video, and MEG signals
- Deep learning-based fusion of multimodal data
- Automatic speech recognition with Whisper
- Contextual understanding using large language models
- Real-time visualization and recording capabilities
- Modular and extensible architecture
- Support for both real-time and file-based inputs

## System Architecture

The system consists of several key components:

1. **Audio Processing** (`audio_preprocessing.py`)
   - Real-time audio stream handling
   - Feature extraction (STFT, mel spectrograms)
   - Noise reduction

2. **Visual Processing** (`visual_feature_extraction.py`)
   - Lip detection and tracking
   - Visual feature extraction using CNN
   - Support for webcam and video file inputs

3. **MEG Processing** (`meg_processing.py`)
   - Real-time MEG data processing
   - Attention state classification
   - Support for file and device inputs

4. **Fusion Model** (`fusion_model.py`)
   - Transformer-based fusion of multimodal features
   - Enhanced audio generation
   - Model training and evaluation

5. **Language Model** (`language_model.py`)
   - ASR using Whisper
   - Contextual understanding with LLM
   - Real-time transcription and response generation

6. **Real-time Pipeline** (`real_time_pipeline.py`)
   - Asynchronous processing of all modalities
   - Real-time visualization
   - Recording and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aysha2016/avse-cognition-ai.git
cd avse-cognition-ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the System

1. Real-time mode (using microphone and webcam):
```bash
python run_avse.py --mode realtime
```

2. File-based mode:
```bash
python run_avse.py --mode file \
    --audio path/to/audio.wav \
    --video path/to/video.mp4 \
    --meg path/to/meg.npy
```

Additional options:
- `--output`: Output directory for recordings (default: "outputs")
- `--no-display`: Disable real-time display
- `--no-record`: Disable recording

### Training the Model

1. Prepare the dataset:
```bash
python prepare_training_data.py \
    --data-dir path/to/raw/data \
    --output-dir path/to/processed/data
```

2. Train the fusion model:
```bash
python train_fusion_model.py \
    --data-dir path/to/processed/data \
    --output-dir path/to/models \
    --batch-size 32 \
    --epochs 100
```

## Data Format

### Input Data

The system expects the following data formats:

1. **Audio**:
   - Sample rate: 16 kHz
   - Format: WAV or raw audio tensor
   - Channels: Mono

2. **Video**:
   - Resolution: 640x480 (configurable)
   - Frame rate: 30 FPS
   - Format: MP4 or raw video frames

3. **MEG**:
   - Sample rate: 1 kHz
   - Channels: 306 (configurable)
   - Format: NumPy array (.npy)

### Training Data

For training, organize your data as follows:

```
raw_data/
├── metadata.json
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── video/
│   ├── sample1.mp4
│   ├── sample2.mp4
│   └── ...
├── meg/
│   ├── sample1.npy
│   ├── sample2.npy
│   └── ...
└── targets/
    ├── sample1.pt
    ├── sample2.pt
    └── ...
```

The `metadata.json` file should contain:
```json
[
    {
        "id": "sample1",
        "audio_path": "audio/sample1.wav",
        "video_path": "video/sample1.mp4",
        "meg_path": "meg/sample1.npy",
        "target_path": "targets/sample1.pt"
    },
    ...
]
```

## Model Architecture

### Fusion Model

The fusion model uses a transformer-based architecture to combine features from all modalities:

1. **Feature Projection**:
   - Audio features → 256-dim
   - Visual features → 256-dim
   - MEG features → 256-dim

2. **Transformer Encoder**:
   - 4 transformer blocks
   - 8 attention heads
   - 256 hidden dimensions
   - Dropout: 0.1

3. **Audio Decoder**:
   - 2-layer MLP
   - ReLU activation
   - Output: Enhanced audio features

### Training

The model is trained to minimize the MSE loss between enhanced and clean audio features. Training parameters:

- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Validation split: 10%
- Early stopping: Yes

    
```
