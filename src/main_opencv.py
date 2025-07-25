#!/usr/bin/env python3
"""
AI Audience Engagement Scanner - OpenCV Only Version
Main application entry point using OpenCV Haar cascades instead of MediaPipe.
Use this version if MediaPipe is not available on your system.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import time
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector_opencv import FaceDetectorOpenCV
from engagement_analyzer_opencv import EngagementAnalyzerOpenCV
from gui_dashboard import EngagementDashboard
from data_manager import DataManager

class EngagementScannerOpenCV:
    def __init__(self, camera_index=0):
        self.root = tk.Tk()
        self.root.title("AI Audience Engagement Scanner (OpenCV)")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.face_detector = FaceDetectorOpenCV()
        self.engagement_analyzer = EngagementAnalyzerOpenCV()
        self.data_manager = DataManager()
        
        # Initialize GUI
        self.dashboard = EngagementDashboard(self.root, self)
        
        # Camera and processing variables
        self.camera = None
        self.camera_index = camera_index  # Allow configurable camera selection
        self.is_running = False
        self.current_session_id = None
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        
    def start_session(self):
        """Start a new engagement scanning session."""
        try:
            # Initialize camera with configurable index
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
                return False
                
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Create new session in database
            self.current_session_id = self.data_manager.create_session()
            
            # Start processing
            self.is_running = True
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._process_video)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.dashboard.update_status("Session started - Analyzing engagement...")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start session: {str(e)}")
            return False
    
    def stop_session(self):
        """Stop the current engagement scanning session."""
        try:
            self.is_running = False
            self.stop_event.set()
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            if self.current_session_id:
                self.data_manager.end_session(self.current_session_id)
                # Auto-export session data
                self._auto_export_session(self.current_session_id)
                self.current_session_id = None
            
            self.dashboard.update_status("Session stopped")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop session: {str(e)}")
            return False
    
    def _process_video(self):
        """Main video processing loop running in separate thread."""
        frame_count = 0
        last_analysis_time = time.time()
        
        while self.is_running and not self.stop_event.is_set():
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Process every 3rd frame for performance (10 FPS analysis)
                if frame_count % 3 == 0:
                    # Detect faces
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Analyze engagement
                    engagement_data = self.engagement_analyzer.analyze_frame(frame, faces)
                    
                    # Update dashboard
                    self.dashboard.update_frame(frame, faces, engagement_data)
                    
                    # Save data every 5 seconds
                    if current_time - last_analysis_time >= 5.0:
                        if self.current_session_id and engagement_data:
                            self.data_manager.save_engagement_data(
                                self.current_session_id, 
                                engagement_data
                            )
                        last_analysis_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in video processing: {e}")
                break
    
    def export_session_data(self):
        """Export session data to CSV. Exports current session if active, otherwise latest session."""
        try:
            session_id_to_export = None
            
            if self.current_session_id:
                # Export current active session
                session_id_to_export = self.current_session_id
                session_type = "current active"
            else:
                # Export latest completed session
                sessions = self.data_manager.get_all_sessions()
                if not sessions.empty:
                    session_id_to_export = sessions.iloc[0]['id']  # Get latest session (first row since ordered DESC)
                    session_type = "latest completed"
                else:
                    messagebox.showwarning("Warning", "No sessions available to export")
                    return
            
            filename = self.data_manager.export_session_data(session_id_to_export)
            messagebox.showinfo("Success", f"Exported {session_type} session data to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def _auto_export_session(self, session_id):
        """Automatically export session data when session ends."""
        try:
            filename = self.data_manager.export_session_data(session_id)
            print(f"Session {session_id} automatically exported to: {filename}")
            self.dashboard.update_status(f"Session data exported to: {filename}")
        except Exception as e:
            print(f"Auto-export failed for session {session_id}: {e}")
    
    def run(self):
        """Start the application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_closing()
    
    def _on_closing(self):
        """Handle application closing."""
        if self.is_running:
            self.stop_session()
        self.root.destroy()

def main():
    """Main function to run the application."""
    import sys
    
    # Parse command line arguments for camera selection
    camera_index = 0  # Default camera
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
            print(f"Using camera index: {camera_index}")
        except ValueError:
            print("Invalid camera index. Using default camera (0)")
            camera_index = 0
    
    print(f"Starting AI Audience Engagement Scanner (OpenCV Version) with camera {camera_index}...")
    
    # Check if specified camera is available
    test_camera = cv2.VideoCapture(camera_index)
    if not test_camera.isOpened():
        print(f"Error: Camera {camera_index} not detected. Please check camera connection and try again.")
        # Try to suggest available cameras
        print("Checking for available cameras...")
        for i in range(5):  # Check first 5 camera indices
            test_cam = cv2.VideoCapture(i)
            if test_cam.isOpened():
                print(f"  Camera {i}: Available")
                test_cam.release()
        return
    test_camera.release()
    
    # Start application with specified camera
    app = EngagementScannerOpenCV(camera_index)
    app.run()

if __name__ == "__main__":
    main()
