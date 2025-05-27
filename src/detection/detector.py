"""
Player and Ball Detection Module

Uses YOLOv8 for detecting players and basketball in video frames.
Supports both pre-trained COCO models and custom basketball-specific models.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from loguru import logger


class PlayerBallDetector:
    """
    Detector for basketball players and ball using YOLOv8.
    
    Detects:
    - Players (person class)
    - Basketball (sports ball class or custom ball class)
    """
    
    def __init__(self, model_path: str, device: torch.device, 
                 conf_threshold: float = 0.5, iou_threshold: float = 0.4):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: PyTorch device (cuda/cpu)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLOv8 model
        logger.info(f"🔍 Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Class mappings (COCO format)
        self.person_class = 0
        self.sports_ball_class = 32
        
        logger.success("✅ Detection model loaded successfully")
    
    def detect(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect players and ball in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Dictionary containing detections:
            {
                'players': [{'bbox': [x1, y1, x2, y2], 'conf': float}, ...],
                'ball': {'bbox': [x1, y1, x2, y2], 'conf': float} or None
            }
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Parse results
        detections = {
            'players': [],
            'ball': None
        }
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get detection data
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Filter by class
                if cls == self.person_class:
                    # Player detection
                    detections['players'].append({
                        'bbox': bbox.tolist(),
                        'conf': conf,
                        'class': 'player'
                    })
                elif cls == self.sports_ball_class:
                    # Ball detection (keep highest confidence)
                    if detections['ball'] is None or conf > detections['ball']['conf']:
                        detections['ball'] = {
                            'bbox': bbox.tolist(),
                            'conf': conf,
                            'class': 'ball'
                        }
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, List]]:
        """
        Detect players and ball in a batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection dictionaries
        """
        batch_detections = []
        
        # Run batch inference
        results = self.model(frames, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        for result in results:
            detections = {
                'players': [],
                'ball': None
            }
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    if cls == self.person_class:
                        detections['players'].append({
                            'bbox': bbox.tolist(),
                            'conf': conf,
                            'class': 'player'
                        })
                    elif cls == self.sports_ball_class:
                        if detections['ball'] is None or conf > detections['ball']['conf']:
                            detections['ball'] = {
                                'bbox': bbox.tolist(),
                                'conf': conf,
                                'class': 'ball'
                            }
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def filter_court_region(self, detections: Dict[str, List], 
                           court_mask: Optional[np.ndarray] = None) -> Dict[str, List]:
        """
        Filter detections to only include those within the court region.
        
        Args:
            detections: Detection results
            court_mask: Binary mask of court region (optional)
            
        Returns:
            Filtered detections
        """
        if court_mask is None:
            return detections
        
        filtered_detections = {
            'players': [],
            'ball': detections['ball']  # Keep ball detection as-is for now
        }
        
        # Filter players
        for player in detections['players']:
            bbox = player['bbox']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Check if player center is within court
            if (0 <= center_y < court_mask.shape[0] and 
                0 <= center_x < court_mask.shape[1] and
                court_mask[center_y, center_x] > 0):
                filtered_detections['players'].append(player)
        
        return filtered_detections
    
    def validate_ball_detection(self, ball_detection: Dict, frame: np.ndarray) -> bool:
        """
        Validate ball detection using color and shape heuristics.
        
        Args:
            ball_detection: Ball detection dictionary
            frame: Input frame
            
        Returns:
            True if detection is likely a basketball
        """
        if ball_detection is None:
            return False
        
        bbox = ball_detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract ball region
        ball_region = frame[y1:y2, x1:x2]
        if ball_region.size == 0:
            return False
        
        # Check aspect ratio (basketball should be roughly circular)
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        if not (0.7 <= aspect_ratio <= 1.3):  # Allow some tolerance
            return False
        
        # Check color (basketball is typically orange/brown)
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Orange color range in HSV
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([25, 255, 255])
        
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_ratio = np.sum(orange_mask > 0) / orange_mask.size
        
        # If significant portion is orange, likely a basketball
        return orange_ratio > 0.1
    
    def get_detection_stats(self, detections: Dict[str, List]) -> Dict[str, int]:
        """
        Get statistics about detections.
        
        Args:
            detections: Detection results
            
        Returns:
            Statistics dictionary
        """
        return {
            'num_players': len(detections['players']),
            'ball_detected': detections['ball'] is not None,
            'total_detections': len(detections['players']) + (1 if detections['ball'] else 0)
        } 