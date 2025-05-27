#!/usr/bin/env python3
"""
Basketball Analytics Pipeline - Main Entry Point

This script processes basketball videos to extract player tracking, action recognition,
and game statistics using computer vision and machine learning.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import click
import yaml
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import BasketballPipeline
from src.utils.config import load_config
from src.utils.video import validate_video_file


@click.command()
@click.option('--input', '-i', 'input_path', required=True, 
              help='Path to input video file')
@click.option('--output', '-o', 'output_dir', required=True,
              help='Output directory for results')
@click.option('--config', '-c', 'config_path', 
              default='configs/default.yaml',
              help='Path to configuration file')
@click.option('--gpu', default=0, help='GPU device ID to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--save-video', is_flag=True, default=True,
              help='Save annotated output video')
@click.option('--save-stats', is_flag=True, default=True,
              help='Save statistics JSON file')
def main(input_path: str, output_dir: str, config_path: str, 
         gpu: int, verbose: bool, save_video: bool, save_stats: bool):
    """
    Basketball Analytics Pipeline
    
    Process basketball videos to extract player tracking, actions, and statistics.
    """
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    logger.info("🏀 Starting Basketball Analytics Pipeline")
    
    try:
        # Validate inputs
        if not validate_video_file(input_path):
            logger.error(f"Invalid video file: {input_path}")
            sys.exit(1)
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        config = load_config(config_path)
        config['gpu_id'] = gpu
        
        logger.info(f"📹 Input video: {input_path}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"⚙️ Config: {config_path}")
        logger.info(f"🖥️ GPU: {gpu}")
        
        # Initialize pipeline
        pipeline = BasketballPipeline(config)
        
        # Process video
        results = pipeline.process_video(
            input_path=input_path,
            output_dir=output_dir,
            save_video=save_video,
            save_stats=save_stats
        )
        
        # Log results summary
        logger.success("✅ Processing completed successfully!")
        logger.info(f"📊 Processed {results['total_frames']} frames")
        logger.info(f"👥 Tracked {results['num_players']} players")
        logger.info(f"🏀 Detected {results['num_events']} events")
        
        if save_video:
            logger.info(f"🎥 Annotated video: {results['output_video']}")
        if save_stats:
            logger.info(f"📈 Statistics file: {results['stats_file']}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 