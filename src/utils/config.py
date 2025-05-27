"""
Configuration utilities for the basketball analytics pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        config = validate_config(config)
        
        logger.info(f"✅ Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set defaults for configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
    """
    # Set default values if missing
    defaults = {
        'gpu_id': 0,
        'detection': {
            'model_path': 'models/yolov8n.pt',
            'conf_threshold': 0.5,
            'iou_threshold': 0.4
        },
        'segmentation': {
            'sam2_model_path': 'models/sam2_hiera_tiny.pt',
            'model_cfg': 'sam2_hiera_t.yaml'
        },
        'tracking': {
            'tracker_type': 'bytetrack',
            'max_disappeared': 30
        },
        'action': {
            'window_size': 16,
            'inference_interval': 8
        },
        'visualization': {
            'draw_bboxes': True,
            'draw_masks': True
        }
    }
    
    # Merge defaults with provided config
    config = merge_configs(defaults, config)
    
    return config


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.
    
    Args:
        default: Default configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise 