"""Aggregate metrics across multiple videos."""

import numpy as np
from typing import Dict, List, Any
import json


class MetricsAggregator:
    """
    Aggregate metrics from multiple videos for experiment-level reporting.
    """
    
    def __init__(self):
        self.video_metrics: Dict[str, Dict[str, float]] = {}
    
    def add_video_metrics(self, video_name: str, metrics: Dict[str, Any]):
        """
        Add metrics for a single video.
        
        Parameters
        ----------
        video_name : str
            Name of the video
        metrics : Dict[str, Any]
            Dictionary of metrics for this video
        """
        self.video_metrics[video_name] = metrics
    
    def compute_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregated metrics across all videos.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'aggregated' and 'per_video' keys
        """
        if not self.video_metrics:
            return {
                'aggregated': {},
                'per_video': {}
            }
        
        # Extract numeric metrics (exclude video_name and other non-numeric fields)
        numeric_metrics = {}
        for video_name, metrics in self.video_metrics.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['video_name']:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute statistics across videos
        aggregated = {}
        for metric_name, values in numeric_metrics.items():
            if values:
                values_array = np.array(values)
                aggregated[f'{metric_name}_avg'] = float(np.mean(values_array))
                aggregated[f'{metric_name}_median'] = float(np.median(values_array))
                aggregated[f'{metric_name}_std'] = float(np.std(values_array))
                aggregated[f'{metric_name}_min'] = float(np.min(values_array))
                aggregated[f'{metric_name}_max'] = float(np.max(values_array))
        
        # Add summary statistics
        aggregated['num_videos'] = len(self.video_metrics)
        
        return {
            'aggregated': aggregated,
            'per_video': self.video_metrics
        }
    
    def get_per_video_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-video metrics dictionary.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping video names to their metrics
        """
        return self.video_metrics
    
    def save_to_json(self, filepath: str):
        """
        Save aggregated metrics to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        metrics = self.compute_aggregated_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_summary_string(self) -> str:
        """
        Get a human-readable summary of aggregated metrics.
        
        Returns
        -------
        str
            Formatted summary string
        """
        metrics = self.compute_aggregated_metrics()
        aggregated = metrics['aggregated']
        
        if not aggregated:
            return "No metrics available"
        
        summary_lines = [
            f"Aggregated Metrics (across {aggregated.get('num_videos', 0)} videos):",
            "=" * 60,
        ]
        
        # Key metrics to display
        key_metrics = [
            'mean_detection_confidence_avg',
            'avg_track_length_avg',
            'track_fragmentation_rate_avg',
            'frame_to_frame_iou_mean_avg',
            'processing_fps_avg',
            'short_track_ratio_avg',
            'total_processing_time_avg'
        ]
        
        for metric in key_metrics:
            if metric in aggregated:
                value = aggregated[metric]
                metric_display = metric.replace('_', ' ').title()
                summary_lines.append(f"{metric_display}: {value:.4f}")
        
        return "\n".join(summary_lines)
