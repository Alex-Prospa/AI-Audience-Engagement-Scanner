"""
Engagement Analysis Module
Analyzes facial features to determine audience engagement levels.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from face_detector import Face

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

class EngagementAnalyzer:
    """Analyzes facial features to determine engagement levels."""
    
    def __init__(self):
        # Eye landmarks indices for MediaPipe face mesh
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouth landmarks for smile detection
        self.MOUTH_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.MOUTH_CORNERS = [61, 291]  # Left and right mouth corners
        
        # Nose tip for head pose estimation
        self.NOSE_TIP_INDEX = 1
        
        # Engagement thresholds
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Below this = eyes closed
        self.SMILE_THRESHOLD = 0.02  # Mouth corner elevation threshold
        self.LAUGH_THRESHOLD = 0.04  # Higher threshold for laugh detection
        self.EYE_CONTACT_ANGLE_THRESHOLD = 15  # degrees from center
        
        # Tracking variables
        self.previous_smile_states = {}
        self.smile_counters = {}
        self.laugh_counters = {}
        
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
            if face.landmarks is None:
                continue
                
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
            eye_contact, alertness = self._analyze_eyes_and_gaze(landmarks, frame.shape)
            face_metrics['eye_contact'] = eye_contact
            face_metrics['alertness'] = alertness
            
            # Analyze smile and laugh
            is_smiling, is_laughing = self._analyze_smile_laugh(landmarks)
            face_metrics['is_smiling'] = is_smiling
            face_metrics['is_laughing'] = is_laughing
            
            # Analyze head pose
            face_metrics['head_pose'] = self._analyze_head_pose(landmarks, frame.shape)
            
        except Exception as e:
            print(f"Error analyzing face {face.id}: {e}")
        
        return face_metrics
    
    def _analyze_eyes_and_gaze(self, landmarks: np.ndarray, frame_shape: Tuple) -> Tuple[bool, float]:
        """Analyze eye openness and gaze direction."""
        try:
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = self._calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self._calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Determine alertness (eyes open)
            alertness = min(1.0, max(0.0, (avg_ear - 0.15) / 0.15))  # Normalize to 0-1
            
            # Estimate gaze direction (simplified)
            eye_contact = self._estimate_eye_contact(landmarks, frame_shape)
            
            return eye_contact, alertness
            
        except Exception as e:
            print(f"Error in eye analysis: {e}")
            return False, 0.0
    
    def _calculate_eye_aspect_ratio(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for drowsiness detection."""
        try:
            if len(eye_indices) < 6:
                return 0.25  # Default value
            
            # Get eye landmarks
            eye_points = landmarks[eye_indices[:6]]  # Use first 6 points for basic EAR
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # Calculate EAR
            if horizontal > 0:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            else:
                ear = 0.25
            
            return ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.25
    
    def _estimate_eye_contact(self, landmarks: np.ndarray, frame_shape: Tuple) -> bool:
        """Estimate if person is making eye contact with camera."""
        try:
            height, width = frame_shape[:2]
            frame_center = np.array([width / 2, height / 2])
            
            # Use nose tip as reference point for head direction
            nose_tip = landmarks[self.NOSE_TIP_INDEX]
            
            # Calculate distance from nose to frame center
            distance_to_center = np.linalg.norm(nose_tip - frame_center)
            
            # Normalize by frame size
            normalized_distance = distance_to_center / (width * 0.3)  # 30% of width as threshold
            
            # Consider eye contact if nose is relatively centered
            return normalized_distance < 1.0
            
        except Exception as e:
            print(f"Error estimating eye contact: {e}")
            return False
    
    def _analyze_smile_laugh(self, landmarks: np.ndarray) -> Tuple[bool, bool]:
        """Analyze mouth shape to detect smiles and laughs."""
        try:
            # Get mouth corner points
            left_corner = landmarks[self.MOUTH_CORNERS[0]]
            right_corner = landmarks[self.MOUTH_CORNERS[1]]
            
            # Get mouth center points for reference
            mouth_points = landmarks[self.MOUTH_INDICES]
            mouth_center = np.mean(mouth_points, axis=0)
            
            # Calculate mouth corner elevation relative to center
            left_elevation = mouth_center[1] - left_corner[1]  # Y increases downward
            right_elevation = mouth_center[1] - right_corner[1]
            
            avg_elevation = (left_elevation + right_elevation) / 2.0
            
            # Calculate mouth width for normalization
            mouth_width = np.linalg.norm(right_corner - left_corner)
            
            if mouth_width > 0:
                normalized_elevation = avg_elevation / mouth_width
            else:
                normalized_elevation = 0
            
            # Determine smile and laugh
            is_smiling = normalized_elevation > self.SMILE_THRESHOLD
            is_laughing = normalized_elevation > self.LAUGH_THRESHOLD
            
            return is_smiling, is_laughing
            
        except Exception as e:
            print(f"Error analyzing smile/laugh: {e}")
            return False, False
    
    def _analyze_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple) -> Tuple[float, float, float]:
        """Estimate head pose (pitch, yaw, roll) from facial landmarks."""
        try:
            # Simplified head pose estimation using key points
            nose_tip = landmarks[self.NOSE_TIP_INDEX]
            
            # Use frame center as reference
            height, width = frame_shape[:2]
            frame_center = np.array([width / 2, height / 2])
            
            # Calculate relative position
            relative_pos = nose_tip - frame_center
            
            # Estimate yaw (left-right head turn)
            yaw = math.atan2(relative_pos[0], width * 0.5) * 180 / math.pi
            
            # Estimate pitch (up-down head tilt)
            pitch = math.atan2(relative_pos[1], height * 0.5) * 180 / math.pi
            
            # Roll estimation would require more complex landmark analysis
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
        """Calculate overall engagement score from individual metrics."""
        if face_count == 0:
            return 0.0
        
        # Weighted combination of metrics
        engagement_score = (
            eye_contact_score * 0.4 +  # 40% weight for eye contact
            alertness_score * 0.3 +    # 30% weight for alertness
            min(100, (smile_count / face_count) * 50) * 0.2 +  # 20% weight for smiles
            min(100, (laugh_count / face_count) * 100) * 0.1   # 10% weight for laughs
        )
        
        return min(100.0, max(0.0, engagement_score))
    
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
