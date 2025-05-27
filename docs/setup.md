# Basketball Analytics Pipeline - Setup Guide

This guide will help you set up the basketball analytics pipeline on your local machine.

## 🔧 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 6GB VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and data

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or 12.x (for GPU acceleration)
- **FFmpeg**: For video processing

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd basketball-analytics
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n basketball-analytics python=3.9
conda activate basketball-analytics

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Install Additional Frameworks

#### MMPose (for pose estimation)
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

#### MMAction2 (for action recognition)
```bash
mim install "mmaction2>=1.1.0"
```

#### SAM2 (for segmentation)
```bash
# Clone and install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### 5. Download Models

```bash
python scripts/download_models.py
```

This will download:
- YOLOv8 models for object detection
- Create placeholders for SAM2 and action models

### 6. Manual Model Downloads

#### SAM2 Models
Download SAM2 checkpoints from the [official repository](https://github.com/facebookresearch/segment-anything-2):

```bash
# Download to models/ directory
wget -O models/sam2_hiera_tiny.pt <sam2_tiny_url>
wget -O models/sam2_hiera_small.pt <sam2_small_url>
```

#### Basketball Action Models
You'll need to train or obtain basketball-specific action recognition models. See the [training guide](training.md) for details.

## 🚀 Quick Start

### Test Installation

```bash
# Test with a sample video
python main.py --input sample_video.mp4 --output results/ --verbose
```

### Basic Usage

```bash
# Process a basketball video
python main.py \
    --input path/to/basketball_video.mp4 \
    --output results/ \
    --config configs/default.yaml \
    --gpu 0
```

### Configuration

Edit `configs/default.yaml` to customize:
- Model paths
- Detection thresholds
- Tracking parameters
- Visualization options

## 📁 Project Structure

```
basketball-analytics/
├── src/                    # Core pipeline modules
│   ├── detection/          # Object detection (YOLOv8)
│   ├── segmentation/       # SAM2 segmentation
│   ├── tracking/           # Multi-object tracking
│   ├── pose/              # Pose estimation
│   ├── action/            # Action classification
│   ├── events/            # Event detection
│   └── utils/             # Utilities
├── models/                # Pre-trained models
├── configs/               # Configuration files
├── data/                  # Training/test data
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size in config
- Use smaller models (YOLOv8n instead of YOLOv8m)
- Enable half-precision: `optimization.half_precision: true`

#### SAM2 Import Error
```bash
# Ensure SAM2 is properly installed
cd segment-anything-2
pip install -e .
```

#### MMPose/MMAction2 Issues
```bash
# Reinstall with mim
mim uninstall mmpose mmaction2
mim install "mmpose>=1.1.0"
mim install "mmaction2>=1.1.0"
```

#### Video Codec Issues
```bash
# Install additional codecs
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

### Performance Optimization

#### For RTX 3060 (6GB VRAM):
```yaml
# In configs/default.yaml
optimization:
  half_precision: true
  batch_size: 1
  frame_skip: 2  # Process every 2nd frame

detection:
  model_path: "models/yolov8n.pt"  # Use nano model

segmentation:
  sam2_model_path: "models/sam2_hiera_tiny.pt"  # Use tiny model
```

#### For Higher-end GPUs:
```yaml
optimization:
  half_precision: true
  batch_size: 4
  frame_skip: 1

detection:
  model_path: "models/yolov8m.pt"  # Use medium model

segmentation:
  sam2_model_path: "models/sam2_hiera_small.pt"  # Use small model
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Test individual components:
```bash
# Test detection
python -m src.detection.detector --test

# Test segmentation
python -m src.segmentation.sam2_segmenter --test
```

## 📊 Monitoring

### GPU Usage
```bash
# Monitor GPU usage during processing
nvidia-smi -l 1
```

### Performance Profiling
```bash
# Enable detailed logging
python main.py --input video.mp4 --output results/ --verbose
```

## 🔄 Updates

Keep the pipeline updated:
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update models
python scripts/download_models.py --force-update
```

## 📞 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review logs in `logs/basketball_analytics.log`
3. Open an issue with:
   - Error message
   - System specifications
   - Configuration used
   - Sample video (if possible)

## 🎯 Next Steps

After setup:
1. [Data Collection Guide](collection.md) - Prepare training data
2. [Training Guide](training.md) - Train custom models
3. [API Reference](api.md) - Detailed API documentation
4. [Examples](examples/) - Sample notebooks and scripts 