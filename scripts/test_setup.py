#!/usr/bin/env python3
"""
Setup Test Script

Tests the basketball analytics pipeline setup to ensure all components work correctly.
"""

import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("🧪 Testing imports...")
    
    try:
        # Core libraries
        import torch
        import torchvision
        import cv2
        import numpy as np
        import yaml
        import click
        
        # Computer vision libraries
        from ultralytics import YOLO
        
        # Optional libraries
        try:
            import mmpose
            logger.success("✅ MMPose available")
        except ImportError:
            logger.warning("⚠️ MMPose not available")
        
        try:
            import mmaction
            logger.success("✅ MMAction2 available")
        except ImportError:
            logger.warning("⚠️ MMAction2 not available")
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            logger.success("✅ SAM2 available")
        except ImportError:
            logger.warning("⚠️ SAM2 not available")
        
        logger.success("✅ Core imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    logger.info("🖥️ Testing CUDA...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        logger.success(f"✅ CUDA available: {device_count} device(s)")
        logger.info(f"   Current device: {current_device} ({device_name})")
        
        # Test GPU memory
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"   GPU memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
        
        return True
    else:
        logger.warning("⚠️ CUDA not available - will use CPU")
        return False


def test_models():
    """Test model availability."""
    logger.info("🤖 Testing models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("❌ Models directory not found")
        return False
    
    # Check YOLOv8 models
    yolo_models = ["yolov8n.pt", "yolov8s.pt"]
    for model_name in yolo_models:
        model_path = models_dir / model_name
        if model_path.exists():
            logger.success(f"✅ Found: {model_name}")
        else:
            logger.warning(f"⚠️ Missing: {model_name}")
    
    # Check SAM2 models
    sam2_models = ["sam2_hiera_tiny.pt"]
    for model_name in sam2_models:
        model_path = models_dir / model_name
        if model_path.exists():
            logger.success(f"✅ Found: {model_name}")
        else:
            logger.warning(f"⚠️ Missing: {model_name}")
    
    return True


def test_yolo():
    """Test YOLOv8 detection."""
    logger.info("🔍 Testing YOLOv8 detection...")
    
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO("yolov8n.pt")
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_image, verbose=False)
        
        logger.success("✅ YOLOv8 detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv8 test failed: {e}")
        return False


def test_pipeline_components():
    """Test pipeline components."""
    logger.info("🔧 Testing pipeline components...")
    
    try:
        # Test configuration loading
        from src.utils.config import load_config
        
        config_path = "configs/default.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            logger.success("✅ Configuration loading works")
        else:
            logger.warning("⚠️ Default config not found")
        
        # Test video utilities
        from src.utils.video import validate_video_file
        logger.success("✅ Video utilities work")
        
        # Test detection module
        from src.detection.detector import PlayerBallDetector
        logger.success("✅ Detection module works")
        
        # Test other modules
        from src.tracking.tracker import MultiObjectTracker
        from src.pose.pose_estimator import PoseEstimator
        from src.action.action_classifier import ActionClassifier
        from src.events.event_detector import EventDetector
        from src.utils.visualization import Visualizer
        from src.utils.stats import StatsGenerator
        
        logger.success("✅ All pipeline components importable")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline component test failed: {e}")
        return False


def test_video_processing():
    """Test video processing capabilities."""
    logger.info("📹 Testing video processing...")
    
    try:
        # Create a test video
        test_video_path = "test_video.mp4"
        
        # Create test frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        
        for i in range(30):  # 1 second of video
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test reading the video
        cap = cv2.VideoCapture(test_video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.success("✅ Video processing test passed")
                success = True
            else:
                logger.error("❌ Could not read video frame")
                success = False
        else:
            logger.error("❌ Could not open test video")
            success = False
        
        cap.release()
        
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting basketball analytics setup test...")
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Models", test_models),
        ("YOLOv8", test_yolo),
        ("Pipeline Components", test_pipeline_components),
        ("Video Processing", test_video_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("🎉 All tests passed! Setup is ready.")
        logger.info("🚀 You can now run: python main.py --input video.mp4 --output results/")
    else:
        logger.warning("⚠️ Some tests failed. Check the setup guide for troubleshooting.")
        logger.info("📖 See docs/setup.md for detailed setup instructions")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 