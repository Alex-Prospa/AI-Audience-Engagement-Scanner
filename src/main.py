#!/usr/bin/env python3
"""
AI Audience Engagement Scanner
Main application entry point for real-time audience engagement analysis.
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

from face_detector import FaceDetector
from engagement_analyzer import EngagementAnalyzer
from gui_dashboard import EngagementDashboard
from data_manager import DataManager

class EngagementScanner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Audience Engagement Scanner")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.engagement_analyzer = EngagementAnalyzer()
        self.data_manager = DataManager()
        
        # Initialize GUI
        self.dashboard = EngagementDashboard(self.root, self)
        
        # Camera and processing variables
        self.camera = None
        self.is_running = False
        self.current_session_id = None
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        
    def start_session(self):
        """Start a new engagement scanning session."""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
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
        """Export current session data to CSV."""
        if not self.current_session_id:
            messagebox.showwarning("Warning", "No active session to export")
            return
        
        try:
            filename = self.data_manager.export_session_data(self.current_session_id)
            messagebox.showinfo("Success", f"Data exported to: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
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
    print("Starting AI Audience Engagement Scanner...")
    
    # Check if camera is available
    test_camera = cv2.VideoCapture(0)
    if not test_camera.isOpened():
        print("Error: No camera detected. Please connect a camera and try again.")
        return
    test_camera.release()
    
    # Start application
    app = EngagementScanner()
    app.run()

if __name__ == "__main__":
    main()
