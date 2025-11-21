"""Collect metrics for a single video during processing."""

from typing import Dict, List, Optional, Tuple
from src.utils.metrics_calculator import (
    calculate_iou,
    normalize_bbox_area,
    compute_track_lengths,
    calculate_short_track_ratio,
    compute_bbox_statistics,
    compute_confidence_statistics,
    calculate_track_fragmentation
)


class VideoMetricsCollector:
    """
    Collect and compute metrics for a single video.
    
    Parameters
    ----------
    video_name : str
        Name of the video file being processed
    """
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        
        # Per-frame data
        self.confidences: List[float] = []
        self.bbox_areas: List[float] = []
        self.frame_count = 0
        
        # Track data
        self.track_history: Dict[int, List[int]] = {}  # track_id -> [frame_numbers]
        self.track_bboxes: Dict[int, Dict[int, List[float]]] = {}  # track_id -> {frame_num -> bbox}
        self.all_track_ids: List[int] = []
        
        # IoU tracking
        self.frame_to_frame_ious: List[float] = []
        
        # Timing
        self.total_processing_time: Optional[float] = None
        self.total_frames: Optional[int] = None
    
    def add_frame_detections(self, detections: List[Dict], frame_shape: Tuple[int, int] = None):
        """
        Add detection data from a single frame.
        
        Parameters
        ----------
        detections : List[Dict]
            List of detection dictionaries with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float
            - 'class_id': int (optional)
        frame_shape : Tuple[int, int], optional
            Frame (height, width) for bbox normalization
        """
        self.frame_count += 1
        
        for det in detections:
            # Store confidence
            self.confidences.append(det['confidence'])
            
            # Normalize and store bbox area if frame shape provided
            if frame_shape is not None:
                height, width = frame_shape
                normalized_area = normalize_bbox_area(det['bbox'], width, height)
                self.bbox_areas.append(normalized_area)
    
    def add_frame_tracks(self, tracks: Dict[int, List[float]], frame_num: int, 
                        previous_tracks: Optional[Dict[int, List[float]]] = None):
        """
        Add tracking data from a single frame.
        
        Parameters
        ----------
        tracks : Dict[int, List[float]]
            Dictionary mapping track_id to bbox [x1, y1, x2, y2]
        frame_num : int
            Current frame number
        previous_tracks : Dict[int, List[float]], optional
            Tracks from previous frame for IoU calculation
        """
        # Update track history
        for track_id, bbox in tracks.items():
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                self.track_bboxes[track_id] = {}
            
            self.track_history[track_id].append(frame_num)
            self.track_bboxes[track_id][frame_num] = bbox
            self.all_track_ids.append(track_id)
        
        # Calculate frame-to-frame IoU for continuing tracks
        if previous_tracks is not None:
            for track_id, bbox in tracks.items():
                if track_id in previous_tracks:
                    iou = calculate_iou(previous_tracks[track_id], bbox)
                    self.frame_to_frame_ious.append(iou)
    
    def set_total_processing_time(self, total_time: float, total_frames: int):
        """
        Set total processing time and frame count.
        
        Parameters
        ----------
        total_time : float
            Total processing time in seconds
        total_frames : int
            Total number of frames processed
        """
        self.total_processing_time = total_time
        self.total_frames = total_frames
    
    def compute_video_metrics(self) -> Dict[str, float]:
        """
        Compute all video-level metrics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Detection confidence metrics
        if self.confidences:
            conf_stats = compute_confidence_statistics(self.confidences)
            metrics['mean_detection_confidence'] = conf_stats['mean']
            metrics['median_detection_confidence'] = conf_stats['median']
            metrics['std_detection_confidence'] = conf_stats['std']
        else:
            metrics['mean_detection_confidence'] = 0.0
            metrics['median_detection_confidence'] = 0.0
            metrics['std_detection_confidence'] = 0.0
        
        # Bounding box size statistics
        if self.bbox_areas:
            bbox_stats = compute_bbox_statistics(self.bbox_areas)
            metrics['bbox_area_mean'] = bbox_stats['mean']
            metrics['bbox_area_median'] = bbox_stats['median']
            metrics['bbox_area_std'] = bbox_stats['std']
        else:
            metrics['bbox_area_mean'] = 0.0
            metrics['bbox_area_median'] = 0.0
            metrics['bbox_area_std'] = 0.0
        
        # Track length metrics
        if self.track_history:
            track_stats = compute_track_lengths(self.track_history)
            metrics['avg_track_length'] = track_stats['mean_length']
            metrics['median_track_length'] = track_stats['median_length']
            metrics['std_track_length'] = track_stats['std_length']
            metrics['total_tracks'] = track_stats['total_tracks']
        else:
            metrics['avg_track_length'] = 0.0
            metrics['median_track_length'] = 0.0
            metrics['std_track_length'] = 0.0
            metrics['total_tracks'] = 0
        
        # Track fragmentation rate
        if self.all_track_ids and self.frame_count > 0:
            metrics['track_fragmentation_rate'] = calculate_track_fragmentation(
                self.all_track_ids, self.frame_count
            )
        else:
            metrics['track_fragmentation_rate'] = 0.0
        
        # Frame-to-frame IoU
        if self.frame_to_frame_ious:
            import numpy as np
            metrics['frame_to_frame_iou_mean'] = float(np.mean(self.frame_to_frame_ious))
            metrics['frame_to_frame_iou_median'] = float(np.median(self.frame_to_frame_ious))
            metrics['frame_to_frame_iou_std'] = float(np.std(self.frame_to_frame_ious))
        else:
            metrics['frame_to_frame_iou_mean'] = 0.0
            metrics['frame_to_frame_iou_median'] = 0.0
            metrics['frame_to_frame_iou_std'] = 0.0
        
        # Short track ratio
        if self.track_history:
            metrics['short_track_ratio'] = calculate_short_track_ratio(self.track_history, threshold=5)
        else:
            metrics['short_track_ratio'] = 0.0
        
        # Processing performance
        if self.total_processing_time is not None and self.total_frames is not None:
            metrics['total_processing_time'] = self.total_processing_time
            metrics['processing_fps'] = (
                self.total_frames / self.total_processing_time 
                if self.total_processing_time > 0 else 0.0
            )
        else:
            metrics['total_processing_time'] = 0.0
            metrics['processing_fps'] = 0.0
        
        # Add video name
        metrics['video_name'] = self.video_name
        metrics['total_frames_processed'] = self.frame_count
        
        return metrics
