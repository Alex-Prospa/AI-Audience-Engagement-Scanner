"""
GUI Dashboard Module
Real-time dashboard for displaying engagement metrics and video feed.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from collections import deque
from typing import List, Optional
# Try to import MediaPipe version first, fallback to OpenCV version
try:
    from face_detector import Face
    from engagement_analyzer import EngagementMetrics
except ImportError:
    from face_detector_opencv import Face
    from engagement_analyzer_opencv import EngagementMetrics

class EngagementDashboard:
    """Real-time GUI dashboard for engagement monitoring."""
    
    def __init__(self, root: tk.Tk, app_controller):
        self.root = root
        self.app_controller = app_controller
        
        # Data storage for trends
        self.engagement_history = deque(maxlen=300)  # 5 minutes at 1 sample/second
        self.time_history = deque(maxlen=300)
        self.session_start_time = None
        
        # GUI update control
        self.last_gui_update = 0
        self.gui_update_interval = 0.1  # 10 FPS for GUI updates
        
        # Initialize GUI components
        self._setup_gui()
        
        # Current frame and metrics
        self.current_frame = None
        self.current_faces = []
        self.current_metrics = None
        
    def _setup_gui(self):
        """Initialize the GUI layout."""
        # Configure main window
        self.root.configure(bg='#2c3e50')
        
        # Create main frames
        self._create_control_frame()
        self._create_video_frame()
        self._create_metrics_frame()
        self._create_status_frame()
        
    def _create_control_frame(self):
        """Create control buttons frame."""
        control_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        control_frame.pack(fill='x', padx=0, pady=0)
        control_frame.pack_propagate(False)
        
        # Add a subtle border
        border_frame = tk.Frame(control_frame, bg='#34495e', height=2)
        border_frame.pack(side='bottom', fill='x')
        
        # Left side - buttons
        button_frame = tk.Frame(control_frame, bg='#2c3e50')
        button_frame.pack(side='left', padx=20, pady=15)
        
        # Start/Stop buttons with improved styling
        self.start_button = tk.Button(
            button_frame, text="▶ Start Session", 
            command=self._start_session,
            bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
            width=15, height=2, relief='flat', bd=0,
            activebackground='#2ecc71', cursor='hand2'
        )
        self.start_button.pack(side='left', padx=8)
        
        self.stop_button = tk.Button(
            button_frame, text="⏹ Stop Session", 
            command=self._stop_session,
            bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
            width=15, height=2, state='disabled', relief='flat', bd=0,
            activebackground='#c0392b', cursor='hand2'
        )
        self.stop_button.pack(side='left', padx=8)
        
        # Export button
        self.export_button = tk.Button(
            button_frame, text="📊 Export Data", 
            command=self._export_data,
            bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
            width=15, height=2, state='normal', relief='flat', bd=0,
            activebackground='#2980b9', cursor='hand2'
        )
        self.export_button.pack(side='left', padx=8)
        
        # Right side - timer and status
        info_frame = tk.Frame(control_frame, bg='#2c3e50')
        info_frame.pack(side='right', padx=20, pady=15)
        
        # Session timer with better styling
        self.timer_label = tk.Label(
            info_frame, text="Session: 00:00:00",
            bg='#2c3e50', fg='#ecf0f1', font=('Arial', 16, 'bold')
        )
        self.timer_label.pack()
        
    def _create_video_frame(self):
        """Create video display frame."""
        video_frame = tk.Frame(self.root, bg='#2c3e50')
        video_frame.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        
        # Video label
        video_title = tk.Label(
            video_frame, text="Live Video Feed",
            bg='#2c3e50', fg='white', font=('Arial', 14, 'bold')
        )
        video_title.pack(pady=5)
        
        # Video canvas
        self.video_canvas = tk.Label(
            video_frame, bg='black', width=640, height=480
        )
        self.video_canvas.pack(pady=5)
        
        # Face count display
        self.face_count_label = tk.Label(
            video_frame, text="Faces Detected: 0",
            bg='#2c3e50', fg='white', font=('Arial', 12)
        )
        self.face_count_label.pack(pady=5)
        
    def _create_metrics_frame(self):
        """Create metrics display frame."""
        # Create right panel container
        right_panel = tk.Frame(self.root, bg='#2c3e50', width=350)
        right_panel.pack(side='right', fill='both', padx=15, pady=5)
        right_panel.pack_propagate(False)
        
        # Metrics frame at the top of right panel
        metrics_frame = tk.Frame(right_panel, bg='#2c3e50')
        metrics_frame.pack(fill='x', pady=(0, 15))
        
        # Metrics title with icon
        title_frame = tk.Frame(metrics_frame, bg='#2c3e50')
        title_frame.pack(fill='x', pady=(10, 15))
        
        metrics_title = tk.Label(
            title_frame, text="📊 Engagement Metrics",
            bg='#2c3e50', fg='#ecf0f1', font=('Arial', 16, 'bold')
        )
        metrics_title.pack()
        
        # Overall engagement with enhanced styling
        self.engagement_frame = tk.Frame(metrics_frame, bg='#34495e', relief='flat', bd=0)
        self.engagement_frame.pack(fill='x', padx=5, pady=(0, 15))
        
        # Add rounded corners effect with padding
        padding_frame = tk.Frame(self.engagement_frame, bg='#34495e', height=10)
        padding_frame.pack(fill='x')
        
        tk.Label(
            self.engagement_frame, text="Overall Engagement",
            bg='#34495e', fg='#bdc3c7', font=('Arial', 12, 'bold')
        ).pack(pady=(10, 5))
        
        self.engagement_value = tk.Label(
            self.engagement_frame, text="0.0%",
            bg='#34495e', fg='#e74c3c', font=('Arial', 32, 'bold')
        )
        self.engagement_value.pack(pady=(0, 15))
        
        # Individual metrics with modern card design
        metrics_data = [
            ("👁 Eye Contact", "eye_contact", "#3498db"),
            ("⚡ Alertness", "alertness", "#f39c12"),
            ("😊 Smiles", "smiles", "#27ae60"),
            ("😂 Laughs", "laughs", "#e67e22")
        ]
        
        self.metric_labels = {}
        for name, key, color in metrics_data:
            # Create card-like frame
            card_frame = tk.Frame(metrics_frame, bg='#34495e', relief='flat', bd=0)
            card_frame.pack(fill='x', padx=5, pady=3)
            
            # Inner frame for padding
            inner_frame = tk.Frame(card_frame, bg='#34495e')
            inner_frame.pack(fill='x', padx=15, pady=12)
            
            # Metric name
            name_label = tk.Label(
                inner_frame, text=name,
                bg='#34495e', fg='#bdc3c7', font=('Arial', 11, 'bold')
            )
            name_label.pack(side='left')
            
            # Metric value
            value_label = tk.Label(
                inner_frame, text="0",
                bg='#34495e', fg=color, font=('Arial', 14, 'bold')
            )
            value_label.pack(side='right')
            
            self.metric_labels[key] = value_label
        
        # Create trend chart underneath metrics in the same right panel
        self._create_trend_chart(right_panel)
    
    def _create_trend_chart(self, parent_frame):
        """Create engagement trend chart in the right panel."""
        # Trend title
        trend_title = tk.Label(
            parent_frame, text="📈 Engagement Trend",
            bg='#2c3e50', fg='#ecf0f1', font=('Arial', 14, 'bold')
        )
        trend_title.pack(pady=(10, 10))
        
        # Create matplotlib figure with compact sizing for right panel
        self.fig = Figure(figsize=(4.5, 3), facecolor='#2c3e50')
        self.ax = self.fig.add_subplot(111, facecolor='#34495e')
        
        # Configure plot appearance for compact display
        self.ax.set_xlabel('Time (s)', color='white', fontsize=9)
        self.ax.set_ylabel('Engagement %', color='white', fontsize=9)
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_ylim(0, 100)
        
        # Create canvas that fills the remaining space in right panel
        self.trend_canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.trend_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=(0, 10))
        
        # Initialize empty plot with better styling
        self.engagement_line, = self.ax.plot([], [], '#27ae60', linewidth=2, label='Overall Engagement', alpha=0.9)
        self.ax.legend(loc='upper right', facecolor='#34495e', edgecolor='white', labelcolor='white', fontsize=8)
        
        # Improve plot margins for compact display
        self.fig.tight_layout(pad=1.0)
        
    def _create_status_frame(self):
        """Create status bar."""
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(side='bottom', fill='x')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame, text="Ready to start session",
            bg='#34495e', fg='white', font=('Arial', 10)
        )
        self.status_label.pack(side='left', padx=10, pady=5)
        
    def _start_session(self):
        """Start engagement monitoring session."""
        if self.app_controller.start_session():
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.export_button.config(state='normal')  # Keep export enabled during session
            
            # Reset data
            self.engagement_history.clear()
            self.time_history.clear()
            self.session_start_time = time.time()
            
            # Start timer update
            self._update_timer()
            
    def _stop_session(self):
        """Stop engagement monitoring session."""
        if self.app_controller.stop_session():
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.export_button.config(state='normal')
            
    def _export_data(self):
        """Export session data."""
        self.app_controller.export_session_data()
        
    def _update_timer(self):
        """Update session timer display."""
        if self.session_start_time and self.stop_button['state'] == 'normal':
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            self.timer_label.config(text=f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.root.after(1000, self._update_timer)
        
    def update_frame(self, frame: np.ndarray, faces: List[Face], metrics: EngagementMetrics):
        """Update video frame and metrics display."""
        current_time = time.time()
        
        # Limit GUI update rate
        if current_time - self.last_gui_update < self.gui_update_interval:
            return
        
        self.last_gui_update = current_time
        
        # Store current data
        self.current_frame = frame
        self.current_faces = faces
        self.current_metrics = metrics
        
        # Update GUI in main thread
        self.root.after_idle(self._update_gui_components)
        
    def _update_gui_components(self):
        """Update GUI components (called in main thread)."""
        try:
            if self.current_frame is not None:
                self._update_video_display()
            
            if self.current_metrics is not None:
                self._update_metrics_display()
                self._update_trend_chart()
                
        except Exception as e:
            print(f"Error updating GUI: {e}")
    
    def _update_video_display(self):
        """Update video display with current frame."""
        try:
            # Draw face annotations - try MediaPipe first, fallback to OpenCV
            try:
                from face_detector import FaceDetector
                detector = FaceDetector()
            except ImportError:
                from face_detector_opencv import FaceDetectorOpenCV
                detector = FaceDetectorOpenCV()
            
            annotated_frame = detector.draw_faces(self.current_frame, self.current_faces)
            
            # Add engagement overlay
            self._add_engagement_overlay(annotated_frame)
            
            # Resize for display
            display_frame = cv2.resize(annotated_frame, (640, 480))
            
            # Convert to PIL Image
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.video_canvas.configure(image=photo)
            self.video_canvas.image = photo  # Keep a reference
            
            # Update face count
            self.face_count_label.config(text=f"Faces Detected: {len(self.current_faces)}")
            
        except Exception as e:
            print(f"Error updating video display: {e}")
    
    def _add_engagement_overlay(self, frame: np.ndarray):
        """Add engagement information overlay to frame."""
        if self.current_metrics is None:
            return
        
        # Add overall engagement score
        engagement_text = f"Engagement: {self.current_metrics.overall_engagement:.1f}%"
        cv2.putText(frame, engagement_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add individual metrics
        y_offset = 90
        metrics_text = [
            f"Eye Contact: {self.current_metrics.eye_contact_score:.1f}%",
            f"Alertness: {self.current_metrics.alertness_score:.1f}%",
            f"Smiles: {self.current_metrics.smile_count}",
            f"Laughs: {self.current_metrics.laugh_count}"
        ]
        
        for text in metrics_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
    
    def _update_metrics_display(self):
        """Update metrics display panel."""
        try:
            metrics = self.current_metrics
            
            # Update overall engagement
            engagement_score = metrics.overall_engagement
            self.engagement_value.config(text=f"{engagement_score:.1f}%")
            
            # Color code engagement level
            if engagement_score > 70:
                color = '#27ae60'  # Green
            elif engagement_score > 40:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            
            self.engagement_value.config(fg=color)
            
            # Update individual metrics
            self.metric_labels['eye_contact'].config(text=f"{metrics.eye_contact_score:.1f}%")
            self.metric_labels['alertness'].config(text=f"{metrics.alertness_score:.1f}%")
            self.metric_labels['smiles'].config(text=str(metrics.smile_count))
            self.metric_labels['laughs'].config(text=str(metrics.laugh_count))
            
        except Exception as e:
            print(f"Error updating metrics display: {e}")
    
    def _update_trend_chart(self):
        """Update engagement trend chart."""
        try:
            if self.session_start_time is None:
                return
            
            # Add current data point
            current_time = time.time() - self.session_start_time
            self.time_history.append(current_time)
            self.engagement_history.append(self.current_metrics.overall_engagement)
            
            # Update plot data
            self.engagement_line.set_data(list(self.time_history), list(self.engagement_history))
            
            # Adjust plot limits
            if self.time_history:
                self.ax.set_xlim(max(0, current_time - 60), current_time + 5)  # Show last 60 seconds
            
            # Redraw canvas
            self.trend_canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating trend chart: {e}")
    
    def update_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)
    
    def cleanup(self):
        """Clean up GUI resources."""
        try:
            if hasattr(self, 'trend_canvas'):
                self.trend_canvas.get_tk_widget().destroy()
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        except Exception as e:
            print(f"Error cleaning up GUI: {e}")
