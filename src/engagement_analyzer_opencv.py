"""
OpenCV-based Engagement Analysis Module
Alternative implementation that works with OpenCV face detection.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from face_detector_opencv import Face

class EngagementMetrics:
    """Container for engagement metrics data."""
    
    def __init__(self):
        self.timestamp = None
        self.total_faces = 0
        self.eye_contact_score = 0.0  # Percentage looking at camera
        self.smile_count = 0
        self.laugh_count = 0
        self.alertness_score = 0.0  # Average alertness across all faces
        self.overall_engagement = 0.0  # Composite engagement score
        self.individual_scores = {}  # Per-face engagement data

class EngagementAnalyzerOpenCV:
    """Analyzes facial features to determine engagement levels using OpenCV."""
    
    def __init__(self):
        # Engagement thresholds
        self.EYE_DISTANCE_THRESHOLD = 0.3  # Relative to face width
        self.SMILE_CONFIDENCE_THRESHOLD = 0.5
        self.EYE_CONTACT_DISTANCE_THRESHOLD = 0.4  # Relative to frame width
        
        # Tracking variables
        self.previous_smile_states = {}
        self.smile_counters = {}
        self.laugh_counters = {}
        self.previous_eye_states = {}
        
    def analyze_frame(self, frame: np.ndarray, faces: List[Face]) -> EngagementMetrics:
        """
        Analyze engagement metrics for all faces in the frame.
        
        Args:
            frame: Input video frame
            faces: List of detected faces
            
        Returns:
            EngagementMetrics object with analysis results
        """
        metrics = EngagementMetrics()
        metrics.timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        metrics.total_faces = len(faces)
        
        if not faces:
            return metrics
        
        eye_contact_count = 0
        total_alertness = 0.0
        total_smiles = 0
        total_laughs = 0
        
        for face in faces:
            # Analyze individual face
            face_metrics = self._analyze_single_face(frame, face)
            metrics.individual_scores[face.id] = face_metrics
            
            # Accumulate metrics
            if face_metrics['eye_contact']:
                eye_contact_count += 1
            
            total_alertness += face_metrics['alertness']
            
            # Track smile/laugh state changes
            self._update_smile_laugh_counters(face.id, face_metrics)
            total_smiles += self.smile_counters.get(face.id, 0)
            total_laughs += self.laugh_counters.get(face.id, 0)
        
        # Calculate aggregate metrics
        metrics.eye_contact_score = (eye_contact_count / len(faces)) * 100
        metrics.alertness_score = (total_alertness / len(faces)) * 100
        metrics.smile_count = total_smiles
        metrics.laugh_count = total_laughs
        
        # Calculate overall engagement score (weighted combination)
        metrics.overall_engagement = self._calculate_overall_engagement(
            metrics.eye_contact_score,
            metrics.alertness_score,
            metrics.smile_count,
            metrics.laugh_count,
            len(faces)
        )
        
        return metrics
    
    def _analyze_single_face(self, frame: np.ndarray, face: Face) -> Dict:
        """Analyze engagement metrics for a single face."""
        landmarks = face.landmarks
        
        # Initialize face metrics
        face_metrics = {
            'eye_contact': False,
            'alertness': 0.0,
            'is_smiling': False,
            'is_laughing': False,
            'head_pose': (0, 0, 0)  # pitch, yaw, roll
        }
        
        try:
            # Analyze eye contact and alertness
            eye_contact, alertness = self._analyze_eyes_and_gaze_opencv(landmarks, face.bbox, frame.shape)
            face_metrics['eye_contact'] = eye_contact
            face_metrics['alertness'] = alertness
            
            # Analyze smile and laugh
            is_smiling, is_laughing = self._analyze_smile_laugh_opencv(landmarks, face.bbox)
            face_metrics['is_smiling'] = is_smiling
            face_metrics['is_laughing'] = is_laughing
            
            # Analyze head pose (simplified)
            face_metrics['head_pose'] = self._analyze_head_pose_opencv(face.bbox, frame.shape)
            
        except Exception as e:
            print(f"Error analyzing face {face.id}: {e}")
        
        return face_metrics
    
    def _analyze_eyes_and_gaze_opencv(self, landmarks: Optional[Dict], bbox: Tuple[int, int, int, int], 
                                     frame_shape: Tuple) -> Tuple[bool, float]:
        """Analyze eye openness and gaze direction using OpenCV features."""
        try:
            x, y, w, h = bbox
            height, width = frame_shape[:2]
            
            # Default values
            alertness = 0.5  # Default moderate alertness
            eye_contact = False
            
            if landmarks and isinstance(landmarks, dict):
                # Check if eyes are detected
                if 'eyes' in landmarks and len(landmarks['eyes']) >= 2:
                    eyes = landmarks['eyes']
                    
                    # Calculate eye distance (indicator of alertness)
                    eye_distance = np.linalg.norm(np.array(eyes[0]) - np.array(eyes[1]))
                    expected_eye_distance = w * self.EYE_DISTANCE_THRESHOLD
                    
                    # Normalize alertness based on eye distance
                    if eye_distance > expected_eye_distance * 0.5:
                        alertness = min(1.0, eye_distance / expected_eye_distance)
                    else:
                        alertness = 0.2  # Eyes likely closed or very drowsy
                    
                    # Estimate eye contact based on face position relative to center
                    face_center = (x + w//2, y + h//2)
                    frame_center = (width//2, height//2)
                    distance_to_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                    
                    # Normalize by frame width
                    normalized_distance = distance_to_center / (width * self.EYE_CONTACT_DISTANCE_THRESHOLD)
                    eye_contact = normalized_distance < 1.0
                else:
                    # No eyes detected, assume moderate alertness
                    alertness = 0.3
                    
                    # Use face position for eye contact estimation
                    face_center = (x + w//2, y + h//2)
                    frame_center = (width//2, height//2)
                    distance_to_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                    normalized_distance = distance_to_center / (width * 0.3)
                    eye_contact = normalized_distance < 1.0
            else:
                # No landmarks, use basic heuristics
                alertness = 0.4  # Assume moderate alertness if face is detected
                
                # Estimate eye contact based on face position
                face_center = (x + w//2, y + h//2)
                frame_center = (width//2, height//2)
                distance_to_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                normalized_distance = distance_to_center / (width * 0.3)
                eye_contact = normalized_distance < 1.0
            
            return eye_contact, alertness
            
        except Exception as e:
            print(f"Error in eye analysis: {e}")
            return False, 0.0
    
    def _analyze_smile_laugh_opencv(self, landmarks: Optional[Dict], bbox: Tuple[int, int, int, int]) -> Tuple[bool, bool]:
        """Analyze smile and laugh using OpenCV smile detection."""
        try:
            is_smiling = False
            is_laughing = False
            
            if landmarks and isinstance(landmarks, dict):
                # Check if smile was detected by OpenCV
                if 'smile_detected' in landmarks:
                    is_smiling = landmarks['smile_detected']
                    
                    # Simple heuristic: if smile is detected and face is large enough, consider it a laugh
                    x, y, w, h = bbox
                    face_area = w * h
                    if is_smiling and face_area > 10000:  # Large face with smile might be laughing
                        is_laughing = True
            
            return is_smiling, is_laughing
            
        except Exception as e:
            print(f"Error analyzing smile/laugh: {e}")
            return False, False
    
    def _analyze_head_pose_opencv(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple) -> Tuple[float, float, float]:
        """Estimate head pose from face bounding box position."""
        try:
            x, y, w, h = bbox
            height, width = frame_shape[:2]
            
            # Calculate face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Calculate relative position
            frame_center_x = width // 2
            frame_center_y = height // 2
            
            # Estimate yaw (left-right head turn) from horizontal position
            yaw = ((face_center_x - frame_center_x) / frame_center_x) * 45  # Max 45 degrees
            
            # Estimate pitch (up-down head tilt) from vertical position
            pitch = ((face_center_y - frame_center_y) / frame_center_y) * 30  # Max 30 degrees
            
            # Roll estimation not available with basic bounding box
            roll = 0.0
            
            return pitch, yaw, roll
            
        except Exception as e:
            print(f"Error analyzing head pose: {e}")
            return 0.0, 0.0, 0.0
    
    def _update_smile_laugh_counters(self, face_id: int, face_metrics: Dict):
        """Update smile and laugh counters for face tracking."""
        current_smiling = face_metrics['is_smiling']
        current_laughing = face_metrics['is_laughing']
        
        # Initialize counters if new face
        if face_id not in self.smile_counters:
            self.smile_counters[face_id] = 0
            self.laugh_counters[face_id] = 0
            self.previous_smile_states[face_id] = {'smiling': False, 'laughing': False}
        
        previous_state = self.previous_smile_states[face_id]
        
        # Count new smiles (transition from not smiling to smiling)
        if current_smiling and not previous_state['smiling']:
            self.smile_counters[face_id] += 1
        
        # Count new laughs (transition from not laughing to laughing)
        if current_laughing and not previous_state['laughing']:
            self.laugh_counters[face_id] += 1
        
        # Update previous state
        self.previous_smile_states[face_id] = {
            'smiling': current_smiling,
            'laughing': current_laughing
        }
    
    def _calculate_overall_engagement(self, eye_contact_score: float, alertness_score: float,
                                    smile_count: int, laugh_count: int, face_count: int) -> float:
        """Calculate overall engagement score from individual metrics with more realistic weighting."""
        if face_count == 0:
            return 0.0
        
        # More sophisticated engagement calculation
        base_engagement = 0.0
        
        # Eye contact component (35% weight) - more strict requirements
        eye_contact_factor = eye_contact_score * 0.35
        
        # Alertness component (40% weight) - most important factor
        # Apply penalties for low alertness
        if alertness_score < 30:
            alertness_factor = alertness_score * 0.2  # Heavy penalty for drowsiness
        elif alertness_score < 60:
            alertness_factor = alertness_score * 0.3  # Moderate penalty
        else:
            alertness_factor = alertness_score * 0.4  # Full weight for high alertness
        
        # Smile/laugh bonus component (25% weight) - but with diminishing returns
        normalized_smiles = min(face_count * 2, smile_count) / (face_count * 2)  # Max 2 smiles per person
        normalized_laughs = min(face_count, laugh_count) / face_count  # Max 1 laugh per person
        
        emotional_engagement = (normalized_smiles * 15 + normalized_laughs * 25) * 0.25
        
        # Base engagement calculation
        base_engagement = eye_contact_factor + alertness_factor + emotional_engagement
        
        # Apply additional factors for more realistic scoring
        
        # Face count penalty (too many faces can reduce individual attention)
        if face_count > 5:
            face_penalty = min(0.8, 1.0 - (face_count - 5) * 0.05)  # Up to 20% penalty
            base_engagement *= face_penalty
        
        # Consistency bonus (reward sustained engagement)
        consistency_bonus = self._calculate_consistency_bonus()
        base_engagement += consistency_bonus
        
        # Apply engagement thresholds for more realistic distribution
        if base_engagement < 20:
            base_engagement *= 0.5  # Make very low engagement even lower
        elif base_engagement > 85:
            # Make very high engagement harder to achieve
            excess = base_engagement - 85
            base_engagement = 85 + (excess * 0.3)
        
        return min(100.0, max(0.0, base_engagement))
    
    def _calculate_consistency_bonus(self) -> float:
        """Calculate bonus points for consistent engagement patterns."""
        # This could be expanded to track engagement over time
        # For now, return a small baseline bonus
        return 2.0  # Small consistent bonus
    
    def reset_counters(self):
        """Reset smile and laugh counters for new session."""
        self.smile_counters.clear()
        self.laugh_counters.clear()
        self.previous_smile_states.clear()
    
    def get_engagement_summary(self, metrics: EngagementMetrics) -> str:
        """Generate a text summary of engagement metrics."""
        if metrics.total_faces == 0:
            return "No faces detected"
        
        engagement_level = "Low"
        if metrics.overall_engagement > 70:
            engagement_level = "High"
        elif metrics.overall_engagement > 40:
            engagement_level = "Medium"
        
        summary = f"""
Engagement Level: {engagement_level} ({metrics.overall_engagement:.1f}%)
Faces Detected: {metrics.total_faces}
Eye Contact: {metrics.eye_contact_score:.1f}%
Alertness: {metrics.alertness_score:.1f}%
Smiles: {metrics.smile_count}
Laughs: {metrics.laugh_count}
        """.strip()
        
        return summary

# Alias for compatibility
EngagementAnalyzer = EngagementAnalyzerOpenCV
