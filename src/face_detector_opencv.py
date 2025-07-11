"""
OpenCV-based Face Detection Module
Alternative implementation using OpenCV Haar cascades instead of MediaPipe.
Use this if MediaPipe is not available on your system.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class Face:
    """Represents a detected face with tracking information."""
    
    def __init__(self, face_id: int, bbox: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None):
        self.id = face_id
        self.bbox = bbox  # (x, y, width, height)
        self.landmarks = landmarks
        self.center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.last_seen = 0
        self.tracking_confidence = 1.0

class FaceDetectorOpenCV:
    """Face detection and tracking using OpenCV Haar cascades."""
    
    def __init__(self):
        # Initialize OpenCV face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Face tracking variables
        self.tracked_faces: Dict[int, Face] = {}
        self.next_face_id = 0
        self.max_tracking_distance = 100  # pixels
        self.max_frames_lost = 10
        
        # Detection parameters - improved for better accuracy
        self.scale_factor = 1.05  # Smaller scale factor for more precise detection
        self.min_neighbors = 8    # Higher value for more reliable detection
        self.min_size = (40, 40)  # Slightly larger minimum size for better quality
        
        # Smoothing parameters for better positioning
        self.position_smoothing = 0.7  # Smoothing factor for bounding box positions
        
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Detect and track faces in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected Face objects
        """
        if frame is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        current_faces = []
        
        for (x, y, w, h) in face_rects:
            bbox = (x, y, w, h)
            
            # Extract face region for feature detection
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes and smile in face region
            landmarks = self._detect_facial_features(face_roi, face_roi_color, bbox)
            
            # Track or create new face
            face = self._track_or_create_face(bbox, landmarks)
            if face:
                current_faces.append(face)
        
        # Update tracking and remove lost faces
        self._update_tracking(current_faces)
        
        return current_faces
    
    def _detect_facial_features(self, face_roi_gray: np.ndarray, face_roi_color: np.ndarray, 
                               bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """Detect facial features within a face region."""
        x, y, w, h = bbox
        features = {}
        
        try:
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
            if len(eyes) >= 2:
                # Convert eye coordinates to global frame coordinates
                eyes_global = []
                for (ex, ey, ew, eh) in eyes:
                    eyes_global.append((x + ex + ew//2, y + ey + eh//2))
                features['eyes'] = eyes_global[:2]  # Take first two eyes
            
            # Detect smile
            smiles = self.smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20)
            features['smile_detected'] = len(smiles) > 0
            if len(smiles) > 0:
                # Convert smile coordinates to global frame coordinates
                sx, sy, sw, sh = smiles[0]
                features['smile_center'] = (x + sx + sw//2, y + sy + sh//2)
            
            # Estimate key facial points
            features['nose_tip'] = (x + w//2, y + int(h*0.6))  # Approximate nose position
            features['mouth_center'] = (x + w//2, y + int(h*0.8))  # Approximate mouth position
            
            return features
            
        except Exception as e:
            print(f"Error detecting facial features: {e}")
            return None
    
    def _track_or_create_face(self, bbox: Tuple[int, int, int, int], 
                             landmarks: Optional[Dict]) -> Optional[Face]:
        """Track existing face or create new one."""
        bbox_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        
        # Try to match with existing tracked faces
        best_match_id = None
        min_distance = float('inf')
        
        for face_id, tracked_face in self.tracked_faces.items():
            distance = np.linalg.norm(np.array(bbox_center) - np.array(tracked_face.center))
            
            if distance < self.max_tracking_distance and distance < min_distance:
                min_distance = distance
                best_match_id = face_id
        
        if best_match_id is not None:
            # Update existing face with position smoothing
            face = self.tracked_faces[best_match_id]
            
            # Apply position smoothing to reduce jitter
            old_bbox = face.bbox
            smoothed_x = int(old_bbox[0] * self.position_smoothing + bbox[0] * (1 - self.position_smoothing))
            smoothed_y = int(old_bbox[1] * self.position_smoothing + bbox[1] * (1 - self.position_smoothing))
            smoothed_w = int(old_bbox[2] * self.position_smoothing + bbox[2] * (1 - self.position_smoothing))
            smoothed_h = int(old_bbox[3] * self.position_smoothing + bbox[3] * (1 - self.position_smoothing))
            
            face.bbox = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
            face.center = (smoothed_x + smoothed_w // 2, smoothed_y + smoothed_h // 2)
            face.last_seen = 0
            face.tracking_confidence = min(1.0, face.tracking_confidence + 0.1)
            if landmarks is not None:
                face.landmarks = landmarks
            return face
        else:
            # Create new face
            new_face = Face(self.next_face_id, bbox, landmarks)
            self.tracked_faces[self.next_face_id] = new_face
            self.next_face_id += 1
            return new_face
    
    def _update_tracking(self, current_faces: List[Face]):
        """Update tracking status and remove lost faces."""
        current_face_ids = {face.id for face in current_faces}
        
        # Update last_seen for faces not detected in current frame
        faces_to_remove = []
        for face_id, face in self.tracked_faces.items():
            if face_id not in current_face_ids:
                face.last_seen += 1
                face.tracking_confidence = max(0.0, face.tracking_confidence - 0.2)
                
                if face.last_seen > self.max_frames_lost:
                    faces_to_remove.append(face_id)
        
        # Remove lost faces
        for face_id in faces_to_remove:
            del self.tracked_faces[face_id]
    
    def get_face_count(self) -> int:
        """Get current number of tracked faces."""
        return len(self.tracked_faces)
    
    def draw_faces(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """
        Draw face bounding boxes and features on the frame.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            
        Returns:
            Frame with drawn face annotations
        """
        annotated_frame = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if face.tracking_confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw face ID
            cv2.putText(annotated_frame, f"ID: {face.id}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw facial features if available
            if face.landmarks and isinstance(face.landmarks, dict):
                # Draw eyes
                if 'eyes' in face.landmarks:
                    for eye_center in face.landmarks['eyes']:
                        cv2.circle(annotated_frame, eye_center, 3, (255, 0, 0), -1)
                
                # Draw nose
                if 'nose_tip' in face.landmarks:
                    cv2.circle(annotated_frame, face.landmarks['nose_tip'], 2, (0, 255, 255), -1)
                
                # Draw mouth
                if 'mouth_center' in face.landmarks:
                    cv2.circle(annotated_frame, face.landmarks['mouth_center'], 2, (255, 255, 0), -1)
                
                # Highlight smile
                if face.landmarks.get('smile_detected', False):
                    cv2.putText(annotated_frame, "SMILE", 
                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw face count
        cv2.putText(annotated_frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame
    
    def cleanup(self):
        """Clean up resources."""
        # OpenCV cascades don't need explicit cleanup
        pass

# Alias for compatibility with main application
FaceDetector = FaceDetectorOpenCV
