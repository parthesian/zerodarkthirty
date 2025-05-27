"""
Basketball Analytics Pipeline

Main pipeline class that orchestrates video processing, object detection,
tracking, action recognition, and event detection.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import torch
from loguru import logger

from .detection.detector import PlayerBallDetector
from .segmentation.sam2_segmenter import SAM2Segmenter
from .tracking.tracker import MultiObjectTracker
from .pose.pose_estimator import PoseEstimator
from .action.action_classifier import ActionClassifier
from .events.event_detector import EventDetector
from .utils.video import VideoProcessor
from .utils.visualization import Visualizer
from .utils.stats import StatsGenerator


class BasketballPipeline:
    """
    Main pipeline for basketball video analysis.
    
    Processes videos through the following stages:
    1. Object Detection (players, ball)
    2. SAM2 Segmentation
    3. Multi-object tracking
    4. Pose estimation
    5. Action classification
    6. Event detection
    7. Statistics generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the basketball analytics pipeline.
        
        Args:
            config: Configuration dictionary containing model paths and parameters
        """
        self.config = config
        self.device = torch.device(f"cuda:{config.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing pipeline components...")
        
        # Object Detection
        self.detector = PlayerBallDetector(
            model_path=self.config['detection']['model_path'],
            device=self.device,
            conf_threshold=self.config['detection']['conf_threshold']
        )
        
        # SAM2 Segmentation
        self.segmenter = SAM2Segmenter(
            model_path=self.config['segmentation']['sam2_model_path'],
            device=self.device
        )
        
        # Multi-object Tracking
        self.tracker = MultiObjectTracker(
            tracker_type=self.config['tracking']['tracker_type'],
            max_disappeared=self.config['tracking']['max_disappeared']
        )
        
        # Pose Estimation
        self.pose_estimator = PoseEstimator(
            model_path=self.config['pose']['model_path'],
            device=self.device
        )
        
        # Action Classification
        self.action_classifier = ActionClassifier(
            model_path=self.config['action']['model_path'],
            device=self.device,
            window_size=self.config['action']['window_size']
        )
        
        # Event Detection
        self.event_detector = EventDetector(
            config=self.config['events']
        )
        
        # Utilities
        self.visualizer = Visualizer(config=self.config['visualization'])
        self.stats_generator = StatsGenerator()
        
        logger.success("✅ All components initialized successfully")
        
    def process_video(self, input_path: str, output_dir: str, 
                     save_video: bool = True, save_stats: bool = True) -> Dict[str, Any]:
        """
        Process a basketball video through the complete pipeline.
        
        Args:
            input_path: Path to input video file
            output_dir: Directory to save outputs
            save_video: Whether to save annotated video
            save_stats: Whether to save statistics JSON
            
        Returns:
            Dictionary containing processing results and output paths
        """
        logger.info(f"🎬 Processing video: {input_path}")
        
        # Initialize video processor
        video_processor = VideoProcessor(input_path)
        
        # Storage for results
        all_detections = []
        all_tracks = []
        all_poses = []
        all_actions = []
        all_events = []
        
        # Video writer for output
        output_video_path = None
        if save_video:
            output_video_path = os.path.join(output_dir, "annotated_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(
                output_video_path, fourcc, 
                video_processor.fps, 
                (video_processor.width, video_processor.height)
            )
        
        frame_count = 0
        start_time = time.time()
        
        try:
            # Process video frame by frame
            for frame in video_processor:
                frame_count += 1
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(f"📊 Processing frame {frame_count}/{video_processor.total_frames}")
                
                # Step 1: Object Detection
                detections = self.detector.detect(frame)
                all_detections.append(detections)
                
                # Step 2: SAM2 Segmentation (if objects detected)
                if detections['players'] or detections['ball']:
                    segmentations = self.segmenter.segment(frame, detections)
                else:
                    segmentations = {'players': [], 'ball': None}
                
                # Step 3: Multi-object Tracking
                tracks = self.tracker.update(detections, segmentations)
                all_tracks.append(tracks)
                
                # Step 4: Pose Estimation (for tracked players)
                poses = {}
                for track_id, track_data in tracks['players'].items():
                    if track_data['bbox'] is not None:
                        pose = self.pose_estimator.estimate(frame, track_data['bbox'])
                        poses[track_id] = pose
                all_poses.append(poses)
                
                # Step 5: Action Classification (every N frames)
                if frame_count % self.config['action']['inference_interval'] == 0:
                    actions = self.action_classifier.classify_batch(
                        tracks['players'], all_poses[-self.config['action']['window_size']:]
                    )
                    all_actions.extend(actions)
                
                # Step 6: Event Detection
                events = self.event_detector.detect_events(
                    frame_count, tracks, poses, all_actions
                )
                all_events.extend(events)
                
                # Step 7: Visualization (if saving video)
                if save_video:
                    annotated_frame = self.visualizer.draw_annotations(
                        frame, detections, tracks, poses, events
                    )
                    out_writer.write(annotated_frame)
                    
        except Exception as e:
            logger.error(f"❌ Error processing frame {frame_count}: {str(e)}")
            raise
            
        finally:
            video_processor.release()
            if save_video and 'out_writer' in locals():
                out_writer.release()
        
        # Processing complete
        processing_time = time.time() - start_time
        fps = frame_count / processing_time
        
        logger.success(f"🎉 Video processing complete!")
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s ({fps:.1f} FPS)")
        
        # Generate statistics
        stats = self.stats_generator.generate_stats(
            all_tracks, all_actions, all_events, video_processor.fps
        )
        
        # Save statistics
        stats_file_path = None
        if save_stats:
            stats_file_path = os.path.join(output_dir, "game_stats.json")
            self.stats_generator.save_stats(stats, stats_file_path)
        
        # Return results
        results = {
            'total_frames': frame_count,
            'processing_time': processing_time,
            'fps': fps,
            'num_players': len(set(track_id for tracks in all_tracks for track_id in tracks['players'].keys())),
            'num_events': len(all_events),
            'output_video': output_video_path,
            'stats_file': stats_file_path,
            'stats': stats
        }
        
        return results
    
    def process_clip(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process a short video clip (for testing or real-time processing).
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Processing results for the clip
        """
        # Similar to process_video but for a list of frames
        # Useful for testing individual components or real-time processing
        pass 