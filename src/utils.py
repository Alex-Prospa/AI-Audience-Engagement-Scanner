"""
Utility Functions
Helper functions for the AI Audience Engagement Scanner.
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional

def resize_frame_for_processing(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Resize frame for faster processing while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width for processing
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame

def calculate_fps(frame_times: List[float], window_size: int = 30) -> float:
    """
    Calculate FPS from frame timestamps.
    
    Args:
        frame_times: List of frame timestamps
        window_size: Number of frames to consider for FPS calculation
        
    Returns:
        Current FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    recent_times = frame_times[-window_size:]
    if len(recent_times) < 2:
        return 0.0
    
    time_diff = recent_times[-1] - recent_times[0]
    if time_diff > 0:
        return (len(recent_times) - 1) / time_diff
    
    return 0.0

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to HH:MM:SS format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def ensure_directory_exists(directory_path: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def validate_camera_index(camera_index: int = 0) -> bool:
    """
    Validate if camera index is available.
    
    Args:
        camera_index: Camera index to test
        
    Returns:
        True if camera is available, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False

def get_available_cameras() -> List[int]:
    """
    Get list of available camera indices.
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    # Test first 5 camera indices
    for i in range(5):
        if validate_camera_index(i):
            available_cameras.append(i)
    
    return available_cameras

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int],
                             font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw text with background rectangle for better visibility.
    
    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        
    Returns:
        Frame with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(frame, 
                 (x - 5, y - text_height - 5),
                 (x + text_width + 5, y + baseline + 5),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    return frame

def calculate_engagement_color(engagement_score: float) -> Tuple[int, int, int]:
    """
    Calculate color based on engagement score.
    
    Args:
        engagement_score: Engagement score (0-100)
        
    Returns:
        BGR color tuple
    """
    if engagement_score >= 70:
        return (0, 255, 0)  # Green
    elif engagement_score >= 40:
        return (0, 165, 255)  # Orange
    else:
        return (0, 0, 255)  # Red

def smooth_values(values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply moving average smoothing to a list of values.
    
    Args:
        values: List of values to smooth
        window_size: Size of smoothing window
        
    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        window_values = values[start_idx:end_idx]
        smoothed.append(sum(window_values) / len(window_values))
    
    return smoothed

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def normalize_landmarks(landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Normalize landmarks relative to bounding box.
    
    Args:
        landmarks: Array of landmark points
        bbox: Bounding box (x, y, width, height)
        
    Returns:
        Normalized landmarks
    """
    x, y, w, h = bbox
    normalized = landmarks.copy()
    
    # Normalize to 0-1 range relative to bounding box
    normalized[:, 0] = (landmarks[:, 0] - x) / w
    normalized[:, 1] = (landmarks[:, 1] - y) / h
    
    return normalized

def create_engagement_summary(metrics_history: List[dict]) -> dict:
    """
    Create summary statistics from engagement metrics history.
    
    Args:
        metrics_history: List of engagement metrics dictionaries
        
    Returns:
        Summary statistics dictionary
    """
    if not metrics_history:
        return {}
    
    # Extract values for each metric
    engagement_scores = [m.get('overall_engagement', 0) for m in metrics_history]
    eye_contact_scores = [m.get('eye_contact_score', 0) for m in metrics_history]
    alertness_scores = [m.get('alertness_score', 0) for m in metrics_history]
    smile_counts = [m.get('smile_count', 0) for m in metrics_history]
    laugh_counts = [m.get('laugh_count', 0) for m in metrics_history]
    
    summary = {
        'total_samples': len(metrics_history),
        'avg_engagement': np.mean(engagement_scores),
        'max_engagement': np.max(engagement_scores),
        'min_engagement': np.min(engagement_scores),
        'std_engagement': np.std(engagement_scores),
        'avg_eye_contact': np.mean(eye_contact_scores),
        'avg_alertness': np.mean(alertness_scores),
        'total_smiles': np.sum(smile_counts),
        'total_laughs': np.sum(laugh_counts),
        'high_engagement_ratio': np.sum(np.array(engagement_scores) > 70) / len(engagement_scores),
        'low_engagement_ratio': np.sum(np.array(engagement_scores) < 40) / len(engagement_scores)
    }
    
    return summary

class PerformanceMonitor:
    """Monitor performance metrics for the application."""
    
    def __init__(self):
        self.frame_times = []
        self.processing_times = []
        self.start_time = None
        
    def start_frame(self):
        """Mark start of frame processing."""
        self.start_time = time.time()
        
    def end_frame(self):
        """Mark end of frame processing and record metrics."""
        if self.start_time:
            current_time = time.time()
            processing_time = current_time - self.start_time
            
            self.frame_times.append(current_time)
            self.processing_times.append(processing_time)
            
            # Keep only recent data (last 100 frames)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
                self.processing_times.pop(0)
            
            self.start_time = None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return calculate_fps(self.frame_times)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per frame."""
        if self.processing_times:
            return np.mean(self.processing_times)
        return 0.0
    
    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        return {
            'fps': self.get_fps(),
            'avg_processing_time': self.get_avg_processing_time(),
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'min_processing_time': np.min(self.processing_times) if self.processing_times else 0,
            'total_frames': len(self.frame_times)
        }

def log_error(error_message: str, error_type: str = "ERROR"):
    """
    Log error message with timestamp.
    
    Args:
        error_message: Error message to log
        error_type: Type of error (ERROR, WARNING, INFO)
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {error_type}: {error_message}")

def validate_engagement_metrics(metrics) -> bool:
    """
    Validate engagement metrics object.
    
    Args:
        metrics: EngagementMetrics object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required attributes
        required_attrs = ['total_faces', 'eye_contact_score', 'alertness_score', 
                         'smile_count', 'laugh_count', 'overall_engagement']
        
        for attr in required_attrs:
            if not hasattr(metrics, attr):
                return False
            
            value = getattr(metrics, attr)
            if value is None:
                return False
        
        # Check value ranges
        if not (0 <= metrics.eye_contact_score <= 100):
            return False
        if not (0 <= metrics.alertness_score <= 100):
            return False
        if not (0 <= metrics.overall_engagement <= 100):
            return False
        if metrics.smile_count < 0 or metrics.laugh_count < 0:
            return False
        if metrics.total_faces < 0:
            return False
            
        return True
        
    except Exception:
        return False
