"""
Data Management Module
Handles database operations and data export for engagement sessions.
"""

import sqlite3
import pandas as pd
import json
import os
import datetime
from typing import Dict, List, Optional, Tuple
# Try to import MediaPipe version first, fallback to OpenCV version
try:
    from engagement_analyzer import EngagementMetrics
except ImportError:
    from engagement_analyzer_opencv import EngagementMetrics

class DataManager:
    """Manages data storage and export for engagement sessions."""
    
    def __init__(self, db_path: str = "data/engagement_data.db"):
        self.db_path = db_path
        self.data_dir = os.path.dirname(db_path)
        
        # Create data directory if it doesn't exist
        if self.data_dir and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Initialize database
        self._init_database()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for handling non-serializable objects."""
        if isinstance(obj, bool):
            return obj
        elif isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return str(obj)
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        duration_seconds INTEGER,
                        total_faces_detected INTEGER DEFAULT 0,
                        avg_engagement_score REAL DEFAULT 0.0,
                        total_smiles INTEGER DEFAULT 0,
                        total_laughs INTEGER DEFAULT 0,
                        notes TEXT
                    )
                ''')
                
                # Create engagement_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS engagement_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        timestamp REAL,
                        total_faces INTEGER,
                        eye_contact_score REAL,
                        alertness_score REAL,
                        smile_count INTEGER,
                        laugh_count INTEGER,
                        overall_engagement REAL,
                        individual_scores TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                ''')
                
                # Create face_tracking table for individual face data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        face_id INTEGER,
                        timestamp REAL,
                        bbox_x INTEGER,
                        bbox_y INTEGER,
                        bbox_width INTEGER,
                        bbox_height INTEGER,
                        eye_contact BOOLEAN,
                        alertness REAL,
                        is_smiling BOOLEAN,
                        is_laughing BOOLEAN,
                        head_pose_pitch REAL,
                        head_pose_yaw REAL,
                        head_pose_roll REAL,
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                ''')
                
                conn.commit()
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def create_session(self) -> int:
        """
        Create a new engagement session.
        
        Returns:
            Session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (start_time) 
                    VALUES (CURRENT_TIMESTAMP)
                ''')
                session_id = cursor.lastrowid
                conn.commit()
                
                print(f"Created new session with ID: {session_id}")
                return session_id
                
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def end_session(self, session_id: int):
        """
        End an engagement session and calculate summary statistics.
        
        Args:
            session_id: ID of the session to end
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate session statistics
                cursor.execute('''
                    SELECT 
                        MIN(timestamp) as start_ts,
                        MAX(timestamp) as end_ts,
                        AVG(overall_engagement) as avg_engagement,
                        MAX(total_faces) as max_faces,
                        MAX(smile_count) as total_smiles,
                        MAX(laugh_count) as total_laughs
                    FROM engagement_data 
                    WHERE session_id = ?
                ''', (session_id,))
                
                stats = cursor.fetchone()
                
                if stats and stats[0] is not None:
                    start_ts, end_ts, avg_engagement, max_faces, total_smiles, total_laughs = stats
                    duration = int(end_ts - start_ts) if end_ts and start_ts else 0
                    
                    # Update session record
                    cursor.execute('''
                        UPDATE sessions 
                        SET end_time = CURRENT_TIMESTAMP,
                            duration_seconds = ?,
                            total_faces_detected = ?,
                            avg_engagement_score = ?,
                            total_smiles = ?,
                            total_laughs = ?
                        WHERE id = ?
                    ''', (duration, max_faces or 0, avg_engagement or 0.0, 
                         total_smiles or 0, total_laughs or 0, session_id))
                
                conn.commit()
                print(f"Session {session_id} ended successfully")
                
        except Exception as e:
            print(f"Error ending session: {e}")
    
    def save_engagement_data(self, session_id: int, metrics: EngagementMetrics):
        """
        Save engagement metrics data to database.
        
        Args:
            session_id: ID of the current session
            metrics: EngagementMetrics object with current data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save aggregate engagement data
                cursor.execute('''
                    INSERT INTO engagement_data (
                        session_id, timestamp, total_faces, eye_contact_score,
                        alertness_score, smile_count, laugh_count, overall_engagement,
                        individual_scores
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    metrics.timestamp,
                    metrics.total_faces,
                    metrics.eye_contact_score,
                    metrics.alertness_score,
                    metrics.smile_count,
                    metrics.laugh_count,
                    metrics.overall_engagement,
                    json.dumps(metrics.individual_scores, default=self._json_serializer)
                ))
                
                # Save individual face tracking data
                for face_id, face_data in metrics.individual_scores.items():
                    cursor.execute('''
                        INSERT INTO face_tracking (
                            session_id, face_id, timestamp, eye_contact, alertness,
                            is_smiling, is_laughing, head_pose_pitch, head_pose_yaw, head_pose_roll
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        face_id,
                        metrics.timestamp,
                        int(face_data.get('eye_contact', False)),  # Convert bool to int
                        float(face_data.get('alertness', 0.0)),
                        int(face_data.get('is_smiling', False)),   # Convert bool to int
                        int(face_data.get('is_laughing', False)),  # Convert bool to int
                        face_data.get('head_pose', (0, 0, 0))[0],  # pitch
                        face_data.get('head_pose', (0, 0, 0))[1],  # yaw
                        face_data.get('head_pose', (0, 0, 0))[2]   # roll
                    ))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error saving engagement data: {e}")
    
    def get_session_data(self, session_id: int) -> Dict:
        """
        Retrieve all data for a specific session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary containing session data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session info
                session_df = pd.read_sql_query('''
                    SELECT * FROM sessions WHERE id = ?
                ''', conn, params=(session_id,))
                
                # Get engagement data
                engagement_df = pd.read_sql_query('''
                    SELECT * FROM engagement_data WHERE session_id = ?
                    ORDER BY timestamp
                ''', conn, params=(session_id,))
                
                # Get face tracking data
                face_tracking_df = pd.read_sql_query('''
                    SELECT * FROM face_tracking WHERE session_id = ?
                    ORDER BY timestamp, face_id
                ''', conn, params=(session_id,))
                
                return {
                    'session_info': session_df,
                    'engagement_data': engagement_df,
                    'face_tracking': face_tracking_df
                }
                
        except Exception as e:
            print(f"Error retrieving session data: {e}")
            return {}
    
    def export_session_data(self, session_id: int) -> str:
        """
        Export session data to CSV files.
        
        Args:
            session_id: ID of the session to export
            
        Returns:
            Path to the exported files directory
        """
        try:
            # Create export directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"data/exports/session_{session_id}_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # Get session data
            session_data = self.get_session_data(session_id)
            
            if not session_data:
                raise Exception("No data found for session")
            
            # Export session summary
            session_info = session_data['session_info']
            session_info.to_csv(f"{export_dir}/session_summary.csv", index=False)
            
            # Export engagement metrics
            engagement_data = session_data['engagement_data']
            if not engagement_data.empty:
                engagement_data.to_csv(f"{export_dir}/engagement_metrics.csv", index=False)
                
                # Create engagement trend chart
                self._create_engagement_chart(engagement_data, export_dir)
            
            # Export face tracking data
            face_tracking = session_data['face_tracking']
            if not face_tracking.empty:
                face_tracking.to_csv(f"{export_dir}/face_tracking.csv", index=False)
            
            # Create summary report
            self._create_summary_report(session_data, export_dir)
            
            print(f"Data exported to: {export_dir}")
            return export_dir
            
        except Exception as e:
            print(f"Error exporting session data: {e}")
            raise
    
    def _create_engagement_chart(self, engagement_data: pd.DataFrame, export_dir: str):
        """Create and save engagement trend chart."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Convert timestamp to relative time
            start_time = engagement_data['timestamp'].min()
            engagement_data['relative_time'] = engagement_data['timestamp'] - start_time
            
            # Plot overall engagement
            plt.subplot(2, 1, 1)
            plt.plot(engagement_data['relative_time'], engagement_data['overall_engagement'], 
                    'g-', linewidth=2, label='Overall Engagement')
            plt.ylabel('Engagement %')
            plt.title('Engagement Trends Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot individual metrics
            plt.subplot(2, 1, 2)
            plt.plot(engagement_data['relative_time'], engagement_data['eye_contact_score'], 
                    'b-', label='Eye Contact %')
            plt.plot(engagement_data['relative_time'], engagement_data['alertness_score'], 
                    'r-', label='Alertness %')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Score %')
            plt.title('Individual Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{export_dir}/engagement_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating engagement chart: {e}")
    
    def _create_summary_report(self, session_data: Dict, export_dir: str):
        """Create a text summary report."""
        try:
            session_info = session_data['session_info'].iloc[0]
            engagement_data = session_data['engagement_data']
            
            report_lines = [
                "AUDIENCE ENGAGEMENT ANALYSIS REPORT",
                "=" * 40,
                "",
                f"Session ID: {session_info['id']}",
                f"Start Time: {session_info['start_time']}",
                f"End Time: {session_info['end_time']}",
                f"Duration: {session_info['duration_seconds']} seconds",
                "",
                "SUMMARY STATISTICS:",
                "-" * 20,
                f"Average Engagement Score: {session_info['avg_engagement_score']:.1f}%",
                f"Maximum Faces Detected: {session_info['total_faces_detected']}",
                f"Total Smiles: {session_info['total_smiles']}",
                f"Total Laughs: {session_info['total_laughs']}",
                ""
            ]
            
            if not engagement_data.empty:
                report_lines.extend([
                    "DETAILED METRICS:",
                    "-" * 20,
                    f"Peak Engagement: {engagement_data['overall_engagement'].max():.1f}%",
                    f"Lowest Engagement: {engagement_data['overall_engagement'].min():.1f}%",
                    f"Average Eye Contact: {engagement_data['eye_contact_score'].mean():.1f}%",
                    f"Average Alertness: {engagement_data['alertness_score'].mean():.1f}%",
                    f"Engagement Standard Deviation: {engagement_data['overall_engagement'].std():.1f}%",
                    ""
                ])
                
                # Engagement level analysis
                high_engagement = (engagement_data['overall_engagement'] > 70).sum()
                medium_engagement = ((engagement_data['overall_engagement'] > 40) & 
                                   (engagement_data['overall_engagement'] <= 70)).sum()
                low_engagement = (engagement_data['overall_engagement'] <= 40).sum()
                total_samples = len(engagement_data)
                
                report_lines.extend([
                    "ENGAGEMENT DISTRIBUTION:",
                    "-" * 20,
                    f"High Engagement (>70%): {high_engagement}/{total_samples} samples ({high_engagement/total_samples*100:.1f}%)",
                    f"Medium Engagement (40-70%): {medium_engagement}/{total_samples} samples ({medium_engagement/total_samples*100:.1f}%)",
                    f"Low Engagement (<40%): {low_engagement}/{total_samples} samples ({low_engagement/total_samples*100:.1f}%)",
                    ""
                ])
            
            # Write report to file
            with open(f"{export_dir}/summary_report.txt", 'w') as f:
                f.write('\n'.join(report_lines))
                
        except Exception as e:
            print(f"Error creating summary report: {e}")
    
    def get_all_sessions(self) -> pd.DataFrame:
        """Get list of all sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query('''
                    SELECT id, start_time, end_time, duration_seconds, 
                           avg_engagement_score, total_faces_detected
                    FROM sessions 
                    ORDER BY start_time DESC
                ''', conn)
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return pd.DataFrame()
    
    def delete_session(self, session_id: int):
        """Delete a session and all its data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete in order due to foreign key constraints
                cursor.execute('DELETE FROM face_tracking WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM engagement_data WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
                
                conn.commit()
                print(f"Session {session_id} deleted successfully")
                
        except Exception as e:
            print(f"Error deleting session: {e}")
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Delete sessions older than specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM face_tracking 
                    WHERE session_id IN (
                        SELECT id FROM sessions 
                        WHERE start_time < datetime('now', '-{} days')
                    )
                '''.format(days_old))
                
                cursor.execute('''
                    DELETE FROM engagement_data 
                    WHERE session_id IN (
                        SELECT id FROM sessions 
                        WHERE start_time < datetime('now', '-{} days')
                    )
                '''.format(days_old))
                
                cursor.execute('''
                    DELETE FROM sessions 
                    WHERE start_time < datetime('now', '-{} days')
                '''.format(days_old))
                
                conn.commit()
                print(f"Cleaned up sessions older than {days_old} days")
                
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
