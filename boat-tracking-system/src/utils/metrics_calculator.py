"""Utility functions for calculating tracking and detection metrics."""

import numpy as np
from typing import Dict, List, Tuple


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Parameters
    ----------
    box1 : List[float]
        First bounding box [x1, y1, x2, y2]
    box2 : List[float]
        Second bounding box [x1, y1, x2, y2]
    
    Returns
    -------
    float
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def normalize_bbox_area(bbox: List[float], frame_width: int, frame_height: int) -> float:
    """
    Normalize bounding box area by frame size.
    
    Parameters
    ----------
    bbox : List[float]
        Bounding box [x1, y1, x2, y2]
    frame_width : int
        Frame width in pixels
    frame_height : int
        Frame height in pixels
    
    Returns
    -------
    float
        Normalized area (0 to 1)
    """
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    
    if frame_area == 0:
        return 0.0
    
    return bbox_area / frame_area


def calculate_track_fragmentation(track_ids: List[int], total_frames: int) -> float:
    """
    Calculate track fragmentation rate.
    
    Parameters
    ----------
    track_ids : List[int]
        List of unique track IDs observed
    total_frames : int
        Total number of frames processed
    
    Returns
    -------
    float
        Fragmentation rate (unique tracks / total frames)
    """
    if total_frames == 0:
        return 0.0
    
    unique_tracks = len(set(track_ids))
    return unique_tracks / total_frames


def compute_track_lengths(track_history: Dict[int, List[int]]) -> Dict[str, float]:
    """
    Compute track length statistics.
    
    Parameters
    ----------
    track_history : Dict[int, List[int]]
        Dictionary mapping track_id to list of frame numbers where it appeared
    
    Returns
    -------
    Dict[str, float]
        Dictionary with track statistics:
        - 'mean_length': Average track length
        - 'median_length': Median track length
        - 'std_length': Standard deviation of track lengths
        - 'min_length': Minimum track length
        - 'max_length': Maximum track length
        - 'total_tracks': Total number of tracks
    """
    if not track_history:
        return {
            'mean_length': 0.0,
            'median_length': 0.0,
            'std_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'total_tracks': 0
        }
    
    track_lengths = [len(frames) for frames in track_history.values()]
    
    return {
        'mean_length': float(np.mean(track_lengths)),
        'median_length': float(np.median(track_lengths)),
        'std_length': float(np.std(track_lengths)),
        'min_length': float(np.min(track_lengths)),
        'max_length': float(np.max(track_lengths)),
        'total_tracks': len(track_lengths)
    }


def calculate_short_track_ratio(track_history: Dict[int, List[int]], threshold: int = 5) -> float:
    """
    Calculate ratio of tracks shorter than threshold.
    
    Parameters
    ----------
    track_history : Dict[int, List[int]]
        Dictionary mapping track_id to list of frame numbers
    threshold : int, optional
        Frame count threshold for "short" tracks (default: 5)
    
    Returns
    -------
    float
        Ratio of short tracks (0 to 1)
    """
    if not track_history:
        return 0.0
    
    track_lengths = [len(frames) for frames in track_history.values()]
    short_tracks = sum(1 for length in track_lengths if length < threshold)
    
    return short_tracks / len(track_lengths)


def compute_bbox_statistics(bbox_areas: List[float]) -> Dict[str, float]:
    """
    Compute bounding box area statistics.
    
    Parameters
    ----------
    bbox_areas : List[float]
        List of normalized bounding box areas
    
    Returns
    -------
    Dict[str, float]
        Dictionary with statistics:
        - 'mean': Mean area
        - 'median': Median area
        - 'std': Standard deviation
        - 'min': Minimum area
        - 'max': Maximum area
    """
    if not bbox_areas:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    areas = np.array(bbox_areas)
    
    return {
        'mean': float(np.mean(areas)),
        'median': float(np.median(areas)),
        'std': float(np.std(areas)),
        'min': float(np.min(areas)),
        'max': float(np.max(areas))
    }


def compute_confidence_statistics(confidences: List[float]) -> Dict[str, float]:
    """
    Compute detection confidence statistics.
    
    Parameters
    ----------
    confidences : List[float]
        List of confidence scores
    
    Returns
    -------
    Dict[str, float]
        Dictionary with statistics:
        - 'mean': Mean confidence
        - 'median': Median confidence
        - 'std': Standard deviation
        - 'min': Minimum confidence
        - 'max': Maximum confidence
    """
    if not confidences:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    confs = np.array(confidences)
    
    return {
        'mean': float(np.mean(confs)),
        'median': float(np.median(confs)),
        'std': float(np.std(confs)),
        'min': float(np.min(confs)),
        'max': float(np.max(confs))
    }
