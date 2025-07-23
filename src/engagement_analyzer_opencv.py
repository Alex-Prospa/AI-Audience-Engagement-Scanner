"""
OpenCV-based Engagement Analysis Module
Alternative implementation that works with OpenCV face detection.
"""

import cv2
import numpy as np
# Consider initializing timestamp with a default value like 0 for consistency
import math
# Consider initializing timestamp with a default value like 0 for consistency
# Consider initializing timestamp with a default value like 0 for consistency
from typing import List, Dict, Optional, Tuple
try:
    from .face_detector_opencv import Face
except ImportError:
    from face_detector_opencv import Face

class EngagementMetrics:
    """Container for engagement metrics data."""
    
    def __init__(self):
        self.timestamp = 0  # Initialize with default value for consistency
        self.total_faces = 0
        
        # Core engagement components (updated weights)
        self.eye_contact_score = 0.0  # Percentage looking at camera (15% weight)
        self.alertness_score = 0.0  # Enhanced alertness detection (35% weight)
        self.attention_quality_score = 0.0  # New: sustained attention metrics (30% weight)
        self.emotional_engagement_score = 0.0  # Enhanced emotional analysis (20% weight)
        
        # Legacy metrics for compatibility
        self.smile_count = 0
        self.laugh_count = 0
        self.overall_engagement = 0.0  # Composite engagement score
        
        # Enhanced metrics
        self.blink_rate = 0.0  # Blinks per minute
        self.attention_span = 0.0  # Average sustained attention duration
        self.engagement_consistency = 0.0  # Consistency bonus/penalty
        self.fidgeting_score = 0.0  # Head movement restlessness
        self.individual_scores = {}  # Per-face engagement data

class EngagementAnalyzerOpenCV:
    """Analyzes facial features to determine engagement levels using OpenCV."""
    
    def __init__(self):
        # Enhanced engagement thresholds (using constants for better readability)
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.25  # For drowsiness detection
        self.BLINK_DURATION_THRESHOLD = 0.3  # Seconds for blink detection
        self.ATTENTION_SPAN_THRESHOLD = 2.0  # Minimum seconds for sustained attention
        self.HEAD_MOVEMENT_THRESHOLD = 15.0  # Degrees for fidgeting detection
        self.EYE_CONTACT_DISTANCE_THRESHOLD = 0.3  # Relative to frame width (reduced for accuracy)
        
        # Legacy tracking variables
        self.previous_smile_states = {}
        self.smile_counters = {}
        self.laugh_counters = {}
        self.previous_eye_states = {}
        
        # Enhanced tracking variables
        self.face_histories = {}  # Store historical data per face
        self.blink_counters = {}  # Track blink frequency
        self.attention_timers = {}  # Track sustained attention
        self.head_pose_histories = {}  # Track head movement patterns
        self.engagement_histories = {}  # Track engagement consistency
        self.frame_count = 0  # For temporal analysis
        
    def analyze_frame(self, frame: np.ndarray, faces: List[Face]) -> EngagementMetrics:
        """
        Analyze engagement metrics for all faces in the frame with enhanced detection.
        
        Args:
            frame: Input video frame
            faces: List of detected faces
            
        Returns:
            EngagementMetrics object with analysis results
        """
        metrics = EngagementMetrics()
        metrics.timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        metrics.total_faces = len(faces)
        self.frame_count += 1
        
        if not faces:
            return metrics
        
        # Enhanced accumulation variables
        eye_contact_count = 0
        total_alertness = 0.0
        total_attention_quality = 0.0
        total_emotional_engagement = 0.0
        total_blink_rate = 0.0
        total_fidgeting = 0.0
        total_attention_span = 0.0
        total_smiles = 0
        total_laughs = 0
        
        for face in faces:
            # Analyze individual face with enhanced metrics
            face_metrics = self._analyze_single_face_enhanced(frame, face)
            metrics.individual_scores[face.id] = face_metrics
            
            # Accumulate enhanced metrics
            if face_metrics['eye_contact']:
                eye_contact_count += 1
            
            total_alertness += face_metrics['alertness']
            total_attention_quality += face_metrics['attention_quality']
            total_emotional_engagement += face_metrics['emotional_engagement']
            total_blink_rate += face_metrics['blink_rate']
            total_fidgeting += face_metrics['fidgeting_score']
            total_attention_span += face_metrics['attention_span']
            
            # Update enhanced tracking
            self._update_enhanced_tracking(face.id, face_metrics)
            total_smiles += self.smile_counters.get(face.id, 0)
            total_laughs += self.laugh_counters.get(face.id, 0)
        
            # Calculate enhanced aggregate metrics
        face_count = len(faces)
        metrics.eye_contact_score = (eye_contact_count / face_count) * 100
        metrics.alertness_score = (total_alertness / face_count) * 100
        metrics.attention_quality_score = total_attention_quality / face_count  # Already in percentage
        metrics.emotional_engagement_score = total_emotional_engagement / face_count  # Already in percentage
        metrics.blink_rate = total_blink_rate / face_count
        metrics.fidgeting_score = total_fidgeting / face_count
        metrics.attention_span = total_attention_span / face_count
        
        # Legacy metrics for compatibility
        metrics.smile_count = total_smiles
        metrics.laugh_count = total_laughs
        
        # Calculate engagement consistency
        metrics.engagement_consistency = self._calculate_enhanced_consistency_bonus()
        
        # Calculate overall engagement score with new enhanced formula
        metrics.overall_engagement = self._calculate_enhanced_engagement_score(metrics)
        
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
# Add error logging here for better debugging instead of just printing the error
            
        except Exception as e:
            print(f"Error analyzing face {face.id}: {e}")
        
        return face_metrics
    
    def _analyze_single_face_enhanced(self, frame: np.ndarray, face: Face) -> Dict:
        """Analyze engagement metrics for a single face with enhanced detection."""
        landmarks = face.landmarks
        
        # Initialize enhanced face metrics
        face_metrics = {
            'eye_contact': False,
            'alertness': 0.0,
            'attention_quality': 0.0,
            'emotional_engagement': 0.0,
            'blink_rate': 0.0,
            'fidgeting_score': 0.0,
            'attention_span': 0.0,
            'is_smiling': False,
            'is_laughing': False,
            'head_pose': (0, 0, 0)  # pitch, yaw, roll
        }
        
        try:
            # Enhanced eye contact and alertness analysis
            eye_contact, alertness = self._analyze_enhanced_alertness(landmarks, face.bbox, frame.shape, face.id)
            face_metrics['eye_contact'] = eye_contact
            face_metrics['alertness'] = alertness
            
            # Enhanced attention quality analysis
            face_metrics['attention_quality'] = self._analyze_attention_quality(face.id, eye_contact, alertness)
            
            # Enhanced emotional engagement analysis
            is_smiling, is_laughing, emotion_intensity = self._analyze_enhanced_emotions(landmarks, face.bbox)
            face_metrics['is_smiling'] = is_smiling
            face_metrics['is_laughing'] = is_laughing
            face_metrics['emotional_engagement'] = emotion_intensity
            
            # Blink rate analysis
            face_metrics['blink_rate'] = self._analyze_blink_rate(landmarks, face.id)
            
            # Head pose and fidgeting analysis
            head_pose = self._analyze_head_pose_opencv(face.bbox, frame.shape)
            face_metrics['head_pose'] = head_pose
            face_metrics['fidgeting_score'] = self._analyze_fidgeting(face.id, head_pose)
            
            # Attention span analysis
            face_metrics['attention_span'] = self._calculate_attention_span(face.id, eye_contact, alertness)
            
        except Exception as e:
            print(f"Error in enhanced face analysis for face {face.id}: {e}")
        
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
# Consider adding type hints for the return value of this method
                    eyes = landmarks['eyes']
                    
                    # Calculate eye distance (indicator of alertness)
                    eye_distance = np.linalg.norm(np.array(eyes[0]) - np.array(eyes[1]))
                    expected_eye_distance = w * 0.3  # Legacy threshold for compatibility
                    
                    # Normalize alertness based on eye distance
                    if eye_distance > expected_eye_distance * 0.5:
                        alertness = min(1.0, eye_distance / expected_eye_distance)
# Add a docstring to explain the purpose of this method
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
    
    def _analyze_enhanced_alertness(self, landmarks: Optional[Dict], bbox: Tuple[int, int, int, int], 
                                   frame_shape: Tuple, face_id: int) -> Tuple[bool, float]:
        """Enhanced alertness analysis with blink detection and eye aspect ratio."""
        try:
            x, y, w, h = bbox
            height, width = frame_shape[:2]
            
            # Default values
            alertness = 0.4
            eye_contact = False
            
            if landmarks and isinstance(landmarks, dict) and 'eyes' in landmarks and len(landmarks['eyes']) >= 2:
                eyes = landmarks['eyes']
                
                # Calculate eye aspect ratio for drowsiness detection
                eye_distance = np.linalg.norm(np.array(eyes[0]) - np.array(eyes[1]))
                face_width = w
                
                # Normalized eye aspect ratio
                ear = eye_distance / face_width if face_width > 0 else 0
                
                # Enhanced alertness calculation
                if ear < self.EYE_ASPECT_RATIO_THRESHOLD:
                    alertness = 0.1  # Very drowsy
                elif ear < self.EYE_ASPECT_RATIO_THRESHOLD * 1.5:
                    alertness = 0.3  # Moderately drowsy  
                elif ear < self.EYE_ASPECT_RATIO_THRESHOLD * 2.0:
                    alertness = 0.6  # Moderate alertness
                else:
                    alertness = 0.9  # High alertness
                
                # Enhanced eye contact detection
                face_center = (x + w//2, y + h//2)
                frame_center = (width//2, height//2)
                distance_to_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                
                # More strict eye contact requirements
                normalized_distance = distance_to_center / (width * self.EYE_CONTACT_DISTANCE_THRESHOLD)
                eye_contact = normalized_distance < 0.8  # Stricter threshold
                
                # Bonus for sustained eye contact
                if face_id in self.attention_timers and eye_contact:
                    sustained_time = self.attention_timers[face_id].get('eye_contact_duration', 0)
                    if sustained_time > self.ATTENTION_SPAN_THRESHOLD:
                        alertness = min(1.0, alertness + 0.1)  # Slight bonus for sustained attention
            else:
                # Fallback for no landmarks
                face_center = (x + w//2, y + h//2)
                frame_center = (width//2, height//2)
                distance_to_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                normalized_distance = distance_to_center / (width * 0.25)
                eye_contact = normalized_distance < 1.0
                alertness = 0.4  # Moderate default
            
            return eye_contact, alertness
            
        except Exception as e:
            print(f"Error in enhanced alertness analysis: {e}")
            return False, 0.0
    
    def _analyze_attention_quality(self, face_id: int, eye_contact: bool, alertness: float) -> float:
        """Analyze quality of attention based on consistency and duration."""
        try:
            # Initialize tracking if new face
            if face_id not in self.attention_timers:
                self.attention_timers[face_id] = {
                    'start_time': self.frame_count,
                    'eye_contact_duration': 0,
                    'alertness_history': [],
                    'consistency_score': 0.0
                }
            
            timer = self.attention_timers[face_id]
            
            # Track attention duration
            if eye_contact and alertness > 0.5:
                timer['eye_contact_duration'] += 1  # Count frames
            else:
                timer['eye_contact_duration'] = max(0, timer['eye_contact_duration'] - 1)  # Decay
            
            # Track alertness history for consistency
            timer['alertness_history'].append(alertness)
            if len(timer['alertness_history']) > 30:  # Keep last 30 frames (1 second at 30fps)
                timer['alertness_history'].pop(0)
            
            # Calculate attention quality score
            duration_score = min(1.0, timer['eye_contact_duration'] / 30.0)  # 1 second = max score
            
            # Calculate consistency from alertness variance
            if len(timer['alertness_history']) > 5:
                variance = np.var(timer['alertness_history'])
                consistency_score = max(0.0, 1.0 - variance * 2)  # Lower variance = higher consistency
            else:
                consistency_score = 0.5
            
            # Combined attention quality (60% duration, 40% consistency)
            attention_quality = (duration_score * 0.6 + consistency_score * 0.4) * 100
            
            return min(100.0, attention_quality)
            
        except Exception as e:
            print(f"Error in attention quality analysis: {e}")
            return 0.0
    
    def _analyze_enhanced_emotions(self, landmarks: Optional[Dict], bbox: Tuple[int, int, int, int]) -> Tuple[bool, bool, float]:
        """Enhanced emotional engagement analysis with intensity levels."""
        try:
            is_smiling = False
            is_laughing = False
            emotion_intensity = 0.0
            
            if landmarks and isinstance(landmarks, dict):
                # Enhanced smile detection
                if 'smile_detected' in landmarks:
                    is_smiling = landmarks['smile_detected']
                    
                    if is_smiling:
                        x, y, w, h = bbox
                        face_area = w * h
                        
                        # Calculate emotion intensity based on face size and smile presence
                        if face_area > 15000:  # Very close face with smile
                            emotion_intensity = 0.9
                            is_laughing = True
                        elif face_area > 10000:  # Close face with smile
                            emotion_intensity = 0.7
                        elif face_area > 5000:  # Medium distance with smile
                            emotion_intensity = 0.5
                        else:  # Distant smile
                            emotion_intensity = 0.3
                    
                    # Additional heuristics for engagement
                    if 'mouth_center' in landmarks:
                        # Could add mouth shape analysis here in the future
                        pass
                
                # Neutral face penalty
                if not is_smiling:
                    emotion_intensity = max(0.0, emotion_intensity - 0.1)
            
            # Convert to percentage
            emotion_intensity *= 100
            
            return is_smiling, is_laughing, emotion_intensity
            
        except Exception as e:
            print(f"Error in enhanced emotion analysis: {e}")
            return False, False, 0.0
    
    def _analyze_blink_rate(self, landmarks: Optional[Dict], face_id: int) -> float:
        """Analyze blink rate for engagement assessment."""
        try:
            # Initialize blink tracking if new face
            if face_id not in self.blink_counters:
                self.blink_counters[face_id] = {
                    'blink_count': 0,
                    'start_frame': self.frame_count,
                    'last_eye_state': 'open'
                }
            
            blink_data = self.blink_counters[face_id]
            
            # Simple blink detection based on eye landmarks availability
            current_eye_state = 'open'
            if landmarks and isinstance(landmarks, dict):
                if 'eyes' in landmarks and len(landmarks['eyes']) >= 2:
                    # Eyes detected - likely open
                    current_eye_state = 'open'
                else:
                    # No eyes detected - possibly closed/blink
                    current_eye_state = 'closed'
            
            # Detect blink (transition from open to closed and back to open)
            if (blink_data['last_eye_state'] == 'open' and current_eye_state == 'closed'):
                blink_data['blink_count'] += 1
            
            blink_data['last_eye_state'] = current_eye_state
            
            # Calculate blinks per minute
            frames_elapsed = max(1, self.frame_count - blink_data['start_frame'])
            minutes_elapsed = frames_elapsed / (30 * 60)  # Assuming 30 FPS
            blinks_per_minute = blink_data['blink_count'] / max(0.1, minutes_elapsed)
            
            # Normal blink rate is 12-20 per minute
            # Lower rates might indicate drowsiness, higher rates might indicate stress
            if 12 <= blinks_per_minute <= 20:
                return blinks_per_minute  # Normal range
            elif blinks_per_minute < 12:
                return max(0, blinks_per_minute - 2)  # Penalty for low blink rate
            else:
                return min(20, blinks_per_minute)  # Cap high blink rates
            
        except Exception as e:
            print(f"Error in blink rate analysis: {e}")
            return 15.0  # Default normal blink rate
    
    def _analyze_fidgeting(self, face_id: int, head_pose: Tuple[float, float, float]) -> float:
        """Analyze head movement patterns to detect fidgeting/restlessness."""
        try:
            # Initialize head pose history if new face
            if face_id not in self.head_pose_histories:
                self.head_pose_histories[face_id] = []
            
            history = self.head_pose_histories[face_id]
            history.append(head_pose)
            
            # Keep last 10 frames for movement analysis
            if len(history) > 10:
                history.pop(0)
            
            if len(history) < 3:
                return 0.0  # Not enough data
            
            # Calculate movement variance
            pitch_variance = np.var([pose[0] for pose in history])
            yaw_variance = np.var([pose[1] for pose in history])
            
            # Calculate fidgeting score (higher variance = more fidgeting)
            total_variance = pitch_variance + yaw_variance
            
            # Normalize fidgeting score (0-100, where 0 is still, 100 is very fidgety)
            fidgeting_score = min(100.0, total_variance / 10.0)
            
            return fidgeting_score
            
        except Exception as e:
            print(f"Error in fidgeting analysis: {e}")
            return 0.0
    
    def _calculate_attention_span(self, face_id: int, eye_contact: bool, alertness: float) -> float:
        """Calculate sustained attention span in seconds."""
        try:
            if face_id not in self.attention_timers:
                return 0.0
            
            timer = self.attention_timers[face_id]
            
            # Calculate sustained attention (both eye contact and alertness required)
            if eye_contact and alertness > 0.6:
                sustained_frames = timer.get('eye_contact_duration', 0)
                # Convert frames to seconds (assuming 30 FPS)
                attention_span_seconds = sustained_frames / 30.0
                return min(60.0, attention_span_seconds)  # Cap at 60 seconds
            else:
                return 0.0
            
        except Exception as e:
            print(f"Error calculating attention span: {e}")
            return 0.0
    
    def _update_enhanced_tracking(self, face_id: int, face_metrics: Dict):
        """Update enhanced tracking data for face."""
        try:
            # Update smile/laugh counters (legacy)
            self._update_smile_laugh_counters(face_id, face_metrics)
            
            # Initialize engagement history if new face
            if face_id not in self.engagement_histories:
                self.engagement_histories[face_id] = {
                    'scores': [],
                    'timestamps': [],
                    'consistency_score': 0.0
                }
            
            history = self.engagement_histories[face_id]
            
            # Calculate individual engagement score for this face
            individual_score = (
                face_metrics['alertness'] * 0.35 +
                face_metrics['attention_quality'] * 0.30 +
                face_metrics['emotional_engagement'] * 0.20 +
                (100 - face_metrics['fidgeting_score']) * 0.15  # Less fidgeting = better
            )
            
            history['scores'].append(individual_score)
            history['timestamps'].append(self.frame_count)
            
            # Keep last 60 frames (2 seconds at 30fps)
            if len(history['scores']) > 60:
                history['scores'].pop(0)
                history['timestamps'].pop(0)
            
        except Exception as e:
            print(f"Error updating enhanced tracking: {e}")
    
    def _calculate_enhanced_consistency_bonus(self) -> float:
        """Calculate consistency bonus based on engagement patterns."""
        try:
            if not self.engagement_histories:
                return 0.0
            
            total_consistency = 0.0
            face_count = 0
            
            for face_id, history in self.engagement_histories.items():
                if len(history['scores']) > 5:  # Need enough data
                    scores = history['scores']
                    
                    # Calculate variance (lower = more consistent)
                    variance = np.var(scores)
                    consistency = max(0.0, 10.0 - variance)  # Convert to 0-10 scale
                    
                    total_consistency += consistency
                    face_count += 1
            
            if face_count > 0:
                avg_consistency = total_consistency / face_count
                return min(10.0, avg_consistency)  # Cap at 10% bonus
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating consistency bonus: {e}")
            return 0.0
    
    def _calculate_enhanced_engagement_score(self, metrics: EngagementMetrics) -> float:
        """Calculate overall engagement score using enhanced formula."""
        try:
            if metrics.total_faces == 0:
                return 0.0
            
            # New weighted formula (totals 100%)
            # Eye contact: 15% (reduced weight, more accurate)
            eye_contact_component = metrics.eye_contact_score * 0.15
            
            # Alertness: 35% (enhanced detection)
            alertness_component = metrics.alertness_score * 0.35
            
            # Attention Quality: 30% (new - sustained attention)
            attention_component = metrics.attention_quality_score * 0.30
            
            # Emotional engagement: 20% (enhanced with intensity)
            emotion_component = metrics.emotional_engagement_score * 0.20
            
            # Base engagement score
            base_score = (eye_contact_component + alertness_component + 
                         attention_component + emotion_component)
            
            # Apply penalties and bonuses
            
            # Fidgeting penalty (up to -10 points for excessive movement)
            fidgeting_penalty = min(10.0, metrics.fidgeting_score * 0.1)
            base_score -= fidgeting_penalty
            
            # Consistency bonus (up to +10 points for sustained engagement)
            consistency_bonus = metrics.engagement_consistency
            base_score += consistency_bonus
            
            # Attention span bonus (up to +5 points for sustained attention)
            attention_bonus = min(5.0, metrics.attention_span * 0.5)
            base_score += attention_bonus
            
            # Blink rate adjustment (normal rate gets slight bonus)
            if 12 <= metrics.blink_rate <= 20:
                base_score += 2.0  # Small bonus for normal blink rate
            elif metrics.blink_rate < 8:
                base_score -= 5.0  # Penalty for very low blink rate (drowsiness)
            
            # Face count scaling (more realistic for group settings)
            if metrics.total_faces > 3:
                # Slight penalty for very crowded scenes (harder to maintain individual attention)
                crowd_penalty = min(15.0, (metrics.total_faces - 3) * 2.0)
                base_score -= crowd_penalty
            
            # Apply realistic score distribution
            if base_score < 15:
                base_score *= 0.3  # Make very low scores even lower
            elif base_score > 80:
                # Make very high scores harder to achieve
                excess = base_score - 80
                base_score = 80 + (excess * 0.4)
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            print(f"Error calculating enhanced engagement score: {e}")
            return 0.0
    
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
# Consider adding type hints for the return value of this method
        """Calculate bonus points for consistent engagement patterns."""
        # This could be expanded to track engagement over time
        # For now, return a small baseline bonus
        return 2.0  # Small consistent bonus
    
    def reset_counters(self):
        """Reset all tracking data for new session."""
        # Legacy counters
        self.smile_counters.clear()
        self.laugh_counters.clear()
        self.previous_smile_states.clear()
        self.previous_eye_states.clear()
        
        # Enhanced tracking data
        self.face_histories.clear()
        self.blink_counters.clear()
        self.attention_timers.clear()
        self.head_pose_histories.clear()
        self.engagement_histories.clear()
        self.frame_count = 0
    
    def get_engagement_summary(self, metrics: EngagementMetrics) -> str:
        """Generate a comprehensive text summary of enhanced engagement metrics."""
        if metrics.total_faces == 0:
            return "No faces detected"
        
        # Enhanced engagement level classification
        if metrics.overall_engagement > 75:
            engagement_level = "Excellent"
        elif metrics.overall_engagement > 60:
            engagement_level = "High"
        elif metrics.overall_engagement > 40:
            engagement_level = "Medium"
        elif metrics.overall_engagement > 25:
            engagement_level = "Low"
        else:
            engagement_level = "Very Low"
        
        # Enhanced summary with new metrics
        summary = f"""
=== ENHANCED ENGAGEMENT ANALYSIS ===
Overall Level: {engagement_level} ({metrics.overall_engagement:.1f}%)
Faces Detected: {metrics.total_faces}

Core Metrics:
  • Eye Contact: {metrics.eye_contact_score:.1f}%
  • Alertness: {metrics.alertness_score:.1f}%
  • Attention Quality: {metrics.attention_quality_score:.1f}%
  • Emotional Engagement: {metrics.emotional_engagement_score:.1f}%

Advanced Metrics:
  • Attention Span: {metrics.attention_span:.1f}s
  • Blink Rate: {metrics.blink_rate:.1f}/min
  • Fidgeting Score: {metrics.fidgeting_score:.1f}%
  • Consistency Bonus: {metrics.engagement_consistency:.1f}%

Legacy Counts:
  • Smiles: {metrics.smile_count}
  • Laughs: {metrics.laugh_count}
        """.strip()
        
        return summary

# Alias for compatibility
EngagementAnalyzer = EngagementAnalyzerOpenCV
# Add a docstring to explain the purpose of this method
# Consider adding type hints for the return value of this method
# Add a docstring to explain the purpose of this method
# Consider adding type hints for the return value of this method
# Add a docstring to explain the purpose of this method
