#!/usr/bin/env python3
"""
Model Download Script

Downloads pre-trained models required for the basketball analytics pipeline.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def download_file(url: str, output_path: str, description: str = ""):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        description: Description for progress bar
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"✅ {description} already exists: {output_path}")
        return
    
    logger.info(f"📥 Downloading {description}: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.success(f"✅ Downloaded: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def download_yolo_models():
    """Download YOLOv8 models."""
    logger.info("🔍 Downloading YOLOv8 models...")
    
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    }
    
    for model_name, url in models.items():
        output_path = f"models/{model_name}"
        download_file(url, output_path, f"YOLOv8 {model_name}")


def download_sam2_models():
    """Download SAM2 models."""
    logger.info("🎭 Downloading SAM2 models...")
    
    # Note: These URLs are placeholders - actual SAM2 models need to be downloaded
    # from the official Meta repository
    logger.warning("⚠️ SAM2 models need to be downloaded manually from:")
    logger.warning("   https://github.com/facebookresearch/segment-anything-2")
    
    # Create placeholder files
    sam2_models = [
        "models/sam2_hiera_tiny.pt",
        "models/sam2_hiera_small.pt",
        "models/sam2_hiera_base_plus.pt",
    ]
    
    for model_path in sam2_models:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            # Create placeholder file
            with open(model_path, 'w') as f:
                f.write("# Placeholder - Download actual SAM2 model from Meta repository\n")
            logger.info(f"📝 Created placeholder: {model_path}")


def download_pose_models():
    """Download pose estimation models."""
    logger.info("🤸 Setting up pose estimation models...")
    
    # Note: MMPose models are typically downloaded automatically
    # when using the framework, but we can prepare the directory
    os.makedirs("models/pose", exist_ok=True)
    
    logger.info("📝 Pose models will be downloaded automatically by MMPose")


def download_action_models():
    """Download action recognition models."""
    logger.info("🎬 Setting up action recognition models...")
    
    # Note: Action models need to be trained or downloaded from MMAction2
    os.makedirs("models/action", exist_ok=True)
    
    # Create placeholder
    placeholder_path = "models/action/action_r2plus1d_basketball.pth"
    if not os.path.exists(placeholder_path):
        with open(placeholder_path, 'w') as f:
            f.write("# Placeholder - Train or download basketball action recognition model\n")
        logger.info(f"📝 Created placeholder: {placeholder_path}")


def create_model_configs():
    """Create model configuration files."""
    logger.info("⚙️ Creating model configuration files...")
    
    # SAM2 config
    sam2_config = """# SAM2 Configuration
model_type: "hiera_tiny"
checkpoint: "models/sam2_hiera_tiny.pt"
device: "cuda"
"""
    
    config_path = "models/sam2_hiera_t.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(sam2_config)
    
    logger.info(f"📝 Created SAM2 config: {config_path}")


def main():
    """Main function to download all models."""
    logger.info("🚀 Starting model download process...")
    
    try:
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Download models
        download_yolo_models()
        download_sam2_models()
        download_pose_models()
        download_action_models()
        
        # Create configs
        create_model_configs()
        
        logger.success("🎉 Model setup completed!")
        logger.info("📋 Next steps:")
        logger.info("   1. Download SAM2 models manually from Meta repository")
        logger.info("   2. Train or download basketball action recognition models")
        logger.info("   3. Run: python main.py --input video.mp4 --output results/")
        
    except Exception as e:
        logger.error(f"❌ Model setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 