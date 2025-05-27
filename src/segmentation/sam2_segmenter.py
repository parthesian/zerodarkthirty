"""
SAM2 Segmentation Module

Uses SAM2 (Segment Anything Model 2) for precise segmentation of players and ball
using bounding box prompts from the detection stage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import cv2
from loguru import logger

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logger.warning("SAM2 not installed. Please install segment-anything-2")
    build_sam2_video_predictor = None
    SAM2ImagePredictor = None


class SAM2Segmenter:
    """
    SAM2-based segmentation for basketball players and ball.
    
    Uses detection bounding boxes as prompts to generate precise segmentation masks.
    Supports both single-frame and video mode with temporal consistency.
    """
    
    def __init__(self, model_path: str, device: torch.device, 
                 model_cfg: str = "sam2_hiera_t.yaml"):
        """
        Initialize SAM2 segmenter.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            device: PyTorch device
            model_cfg: SAM2 model configuration file
        """
        self.device = device
        self.model_path = model_path
        self.model_cfg = model_cfg
        
        if build_sam2_video_predictor is None:
            logger.error("SAM2 not available. Segmentation will be disabled.")
            self.predictor = None
            return
        
        # Initialize SAM2 predictor
        logger.info(f"🎭 Loading SAM2 model: {model_path}")
        try:
            self.predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
            self.image_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
            logger.success("✅ SAM2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            self.predictor = None
            self.image_predictor = None
        
        # Video state for temporal consistency
        self.video_state = None
        self.frame_idx = 0
        self.object_ids = {}  # Map detection to SAM2 object IDs
        
    def segment(self, frame: np.ndarray, detections: Dict[str, List]) -> Dict[str, List]:
        """
        Segment objects in a single frame using detection prompts.
        
        Args:
            frame: Input frame as numpy array
            detections: Detection results with bounding boxes
            
        Returns:
            Dictionary containing segmentation masks:
            {
                'players': [{'mask': np.ndarray, 'bbox': list, 'object_id': int}, ...],
                'ball': {'mask': np.ndarray, 'bbox': list, 'object_id': int} or None
            }
        """
        if self.predictor is None:
            return self._fallback_segmentation(detections)
        
        segmentations = {
            'players': [],
            'ball': None
        }
        
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for prediction
            self.image_predictor.set_image(frame_rgb)
            
            # Segment players
            for i, player in enumerate(detections['players']):
                bbox = np.array(player['bbox'])  # [x1, y1, x2, y2]
                
                # Generate mask using bbox prompt
                masks, scores, _ = self.image_predictor.predict(
                    box=bbox[None, :],  # Add batch dimension
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    mask = masks[0]  # Take first (and only) mask
                    segmentations['players'].append({
                        'mask': mask,
                        'bbox': player['bbox'],
                        'conf': player['conf'],
                        'object_id': i,
                        'score': float(scores[0])
                    })
            
            # Segment ball
            if detections['ball'] is not None:
                bbox = np.array(detections['ball']['bbox'])
                
                masks, scores, _ = self.image_predictor.predict(
                    box=bbox[None, :],
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    mask = masks[0]
                    segmentations['ball'] = {
                        'mask': mask,
                        'bbox': detections['ball']['bbox'],
                        'conf': detections['ball']['conf'],
                        'object_id': 999,  # Special ID for ball
                        'score': float(scores[0])
                    }
                    
        except Exception as e:
            logger.warning(f"SAM2 segmentation failed: {e}. Using fallback.")
            return self._fallback_segmentation(detections)
        
        return segmentations
    
    def segment_video_frame(self, frame: np.ndarray, detections: Dict[str, List], 
                           frame_idx: int) -> Dict[str, List]:
        """
        Segment objects in video mode with temporal consistency.
        
        Args:
            frame: Input frame
            detections: Detection results
            frame_idx: Current frame index
            
        Returns:
            Segmentation results with temporal consistency
        """
        if self.predictor is None:
            return self._fallback_segmentation(detections)
        
        # Initialize video state on first frame
        if self.video_state is None:
            self._init_video_state(frame)
        
        # Add prompts for new objects
        self._add_video_prompts(frame_idx, detections)
        
        # Propagate masks
        try:
            out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(self.video_state)
            
            # Convert to segmentation format
            segmentations = self._parse_video_masks(out_obj_ids, out_mask_logits, detections)
            
        except Exception as e:
            logger.warning(f"Video segmentation failed: {e}")
            segmentations = self._fallback_segmentation(detections)
        
        self.frame_idx = frame_idx
        return segmentations
    
    def _init_video_state(self, first_frame: np.ndarray):
        """Initialize video state for SAM2 video predictor."""
        if self.predictor is None:
            return
            
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        self.video_state = self.predictor.init_state(frame_rgb)
        self.frame_idx = 0
        self.object_ids = {}
    
    def _add_video_prompts(self, frame_idx: int, detections: Dict[str, List]):
        """Add prompts for detected objects in video mode."""
        if self.predictor is None or self.video_state is None:
            return
        
        # Add player prompts
        for i, player in enumerate(detections['players']):
            obj_id = i + 1  # Start from 1
            bbox = np.array(player['bbox'])
            
            # Add prompt if new object or re-detection
            if obj_id not in self.object_ids or frame_idx % 30 == 0:  # Re-prompt every 30 frames
                self.predictor.add_new_points_or_box(
                    inference_state=self.video_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=bbox
                )
                self.object_ids[obj_id] = 'player'
        
        # Add ball prompt
        if detections['ball'] is not None:
            obj_id = 999  # Special ID for ball
            bbox = np.array(detections['ball']['bbox'])
            
            if obj_id not in self.object_ids or frame_idx % 15 == 0:  # Re-prompt ball more frequently
                self.predictor.add_new_points_or_box(
                    inference_state=self.video_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=bbox
                )
                self.object_ids[obj_id] = 'ball'
    
    def _parse_video_masks(self, obj_ids: List[int], mask_logits: torch.Tensor, 
                          detections: Dict[str, List]) -> Dict[str, List]:
        """Parse SAM2 video output into segmentation format."""
        segmentations = {
            'players': [],
            'ball': None
        }
        
        for i, obj_id in enumerate(obj_ids):
            mask = (mask_logits[i] > 0.0).cpu().numpy()
            
            if obj_id == 999:  # Ball
                if detections['ball'] is not None:
                    segmentations['ball'] = {
                        'mask': mask,
                        'bbox': detections['ball']['bbox'],
                        'conf': detections['ball']['conf'],
                        'object_id': obj_id,
                        'score': 1.0
                    }
            else:  # Player
                # Match with detection by object ID
                if obj_id - 1 < len(detections['players']):
                    player = detections['players'][obj_id - 1]
                    segmentations['players'].append({
                        'mask': mask,
                        'bbox': player['bbox'],
                        'conf': player['conf'],
                        'object_id': obj_id,
                        'score': 1.0
                    })
        
        return segmentations
    
    def _fallback_segmentation(self, detections: Dict[str, List]) -> Dict[str, List]:
        """
        Fallback segmentation using bounding boxes when SAM2 is not available.
        
        Args:
            detections: Detection results
            
        Returns:
            Segmentation results using bbox masks
        """
        segmentations = {
            'players': [],
            'ball': None
        }
        
        # Create bbox masks for players
        for i, player in enumerate(detections['players']):
            bbox = player['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Create binary mask from bbox
            mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            mask[:, :] = True  # Fill entire bbox
            
            segmentations['players'].append({
                'mask': mask,
                'bbox': bbox,
                'conf': player['conf'],
                'object_id': i,
                'score': 1.0
            })
        
        # Create bbox mask for ball
        if detections['ball'] is not None:
            bbox = detections['ball']['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            mask[:, :] = True
            
            segmentations['ball'] = {
                'mask': mask,
                'bbox': bbox,
                'conf': detections['ball']['conf'],
                'object_id': 999,
                'score': 1.0
            }
        
        return segmentations
    
    def reset_video_state(self):
        """Reset video state for processing a new video."""
        self.video_state = None
        self.frame_idx = 0
        self.object_ids = {}
    
    def get_mask_area(self, mask: np.ndarray) -> int:
        """Get the area (number of pixels) of a mask."""
        return int(np.sum(mask))
    
    def get_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get the centroid of a mask."""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            return centroid_x, centroid_y
        return 0, 0 