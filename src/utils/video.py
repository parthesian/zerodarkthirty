"""
Video processing utilities for the basketball analytics pipeline.
"""

import os
from typing import Iterator, Tuple, Optional
import cv2
import numpy as np
from loguru import logger


def validate_video_file(video_path: str) -> bool:
    """
    Validate if the video file exists and can be opened.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return False
    
    # Check if video has frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        logger.error(f"Video has no frames: {video_path}")
        cap.release()
        return False
    
    cap.release()
    return True


class VideoProcessor:
    """
    Video processor for reading frames from video files.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📹 Video loaded: {video_path}")
        logger.info(f"   Resolution: {self.width}x{self.height}")
        logger.info(f"   FPS: {self.fps}")
        logger.info(f"   Total frames: {self.total_frames}")
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate through video frames.
        
        Yields:
            Video frames as numpy arrays
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame.
        
        Returns:
            Tuple of (success, frame)
        """
        return self.cap.read()
    
    def seek_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if successful
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def get_current_frame_number(self) -> int:
        """Get current frame number."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


class VideoWriter:
    """
    Video writer for saving processed videos.
    """
    
    def __init__(self, output_path: str, fps: float, width: int, height: int, 
                 codec: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            width: Video width
            height: Video height
            codec: Video codec
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")
        
        logger.info(f"📹 Video writer created: {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write
        """
        # Resize frame if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"✅ Video saved: {self.output_path}")


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize frame to target size.
    
    Args:
        frame: Input frame
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if maintain_aspect:
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize and pad
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded frame
        padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    else:
        return cv2.resize(frame, target_size)


def extract_frames(video_path: str, output_dir: str, 
                   frame_interval: int = 1, max_frames: Optional[int] = None) -> int:
    """
    Extract frames from video to directory.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        
    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    logger.info(f"📸 Extracted {extracted_count} frames to {output_dir}")
    return extracted_count


def get_video_info(video_path: str) -> dict:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info 