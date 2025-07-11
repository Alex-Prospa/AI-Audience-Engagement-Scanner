"""
Face Detection Module
Uses MediaPipe for efficient multi-face detection and tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional

class Face:
    """Represents a detected face with tracking information."""
    
    def __init__(self, face_id: int, bbox: Tuple[int, int, int, int], landmarks: np.ndarray):
        self.id = face_id
        self.bbox = bbox  # (x, y, width, height)
        self.landmarks = landmarks
        self.center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.last_seen = 0
        self.tracking_confidence = 1.0

class FaceDetector:
    """Face detection and tracking using MediaPipe."""
    
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for long-range detection (better for audiences)
            min_detection_confidence=0.7
        )
        
        # Face mesh for detailed landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=50,  # Support up to 50 faces
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Face tracking variables
        self.tracked_faces: Dict[int, Face] = {}
        self.next_face_id = 0
        self.max_tracking_distance = 100  # pixels
        self.max_frames_lost = 10
        
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
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Detect faces
        detection_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        current_faces = []
        
        if detection_results.detections:
            for detection in detection_results.detections:
                # Get bounding box
                bbox = self._get_bbox_from_detection(detection, width, height)
                
                # Get landmarks if available
                landmarks = self._get_landmarks_for_bbox(mesh_results, bbox, width, height)
                
                # Track or create new face
                face = self._track_or_create_face(bbox, landmarks)
                if face:
                    current_faces.append(face)
        
        # Update tracking and remove lost faces
        self._update_tracking(current_faces)
        
        return current_faces
    
    def _get_bbox_from_detection(self, detection, width: int, height: int) -> Tuple[int, int, int, int]:
        """Extract bounding box from MediaPipe detection."""
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        
        # Ensure bbox is within frame bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        return (x, y, w, h)
    
    def _get_landmarks_for_bbox(self, mesh_results, bbox: Tuple[int, int, int, int], 
                               width: int, height: int) -> Optional[np.ndarray]:
        """Find face mesh landmarks that correspond to the detected face bbox."""
        if not mesh_results.multi_face_landmarks:
            return None
        
        bbox_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        best_landmarks = None
        min_distance = float('inf')
        
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Calculate center of landmarks
            landmarks_array = np.array([[lm.x * width, lm.y * height] 
                                      for lm in face_landmarks.landmark])
            landmarks_center = np.mean(landmarks_array, axis=0)
            
            # Check if landmarks center is close to bbox center
            distance = np.linalg.norm(np.array(bbox_center) - landmarks_center)
            
            if distance < min_distance and distance < bbox[2] * 0.5:  # Within half bbox width
                min_distance = distance
                best_landmarks = landmarks_array
        
        return best_landmarks
    
    def _track_or_create_face(self, bbox: Tuple[int, int, int, int], 
                             landmarks: Optional[np.ndarray]) -> Optional[Face]:
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
            # Update existing face
            face = self.tracked_faces[best_match_id]
            face.bbox = bbox
            face.center = bbox_center
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
        Draw face bounding boxes and IDs on the frame.
        
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
            
            # Draw landmarks if available
            if face.landmarks is not None:
                for point in face.landmarks[::10]:  # Draw every 10th landmark
                    cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
        
        # Draw face count
        cv2.putText(annotated_frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
