# AI Audience Engagement Scanner

A real-time Python application that analyzes audience engagement through facial recognition and computer vision. The system tracks eye contact, smiles/laughs, and alertness levels to provide comprehensive engagement metrics for presentations, meetings, and events.

## Features

### Real-time Analysis
- **Multi-face Detection**: Simultaneously track up to 50 faces using MediaPipe
- **Eye Contact Tracking**: Detect when audience members are looking at the camera/presenter
- **Smile & Laugh Detection**: Count and track positive emotional responses
- **Alertness Monitoring**: Measure audience attention through eye openness analysis
- **Overall Engagement Score**: Composite metric combining all factors

### Professional Dashboard
- **Live Video Feed**: Real-time camera feed with face tracking overlays
- **Engagement Metrics Panel**: Current statistics and scores
- **Trend Visualization**: Real-time engagement trends over time
- **Session Timer**: Track presentation duration
- **Export Functionality**: Save session data and reports

### Data Management
- **SQLite Database**: Persistent storage of all session data
- **CSV Export**: Export engagement data for further analysis
- **Detailed Reports**: Comprehensive session summaries with statistics
- **Trend Charts**: Visual engagement analysis over time

## Requirements

### Hardware
- **Camera**: Standard webcam (720p minimum recommended)
- **RAM**: 8GB minimum for processing 20-50 faces
- **CPU**: Multi-core processor recommended
- **OS**: Windows, macOS, or Linux

### Software
- Python 3.8 or higher
- See `requirements.txt` for Python dependencies

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alexmoses/AI-Audience-Engagement-Scanner.git
   cd AI-Audience-Engagement-Scanner
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **If you encounter dependency issues**, try the minimal requirements:
   ```bash
   pip install -r requirements-minimal.txt
   ```

4. **Verify camera access**:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
   ```

5. **Detect available cameras** (if you have multiple cameras):
   ```bash
   python detect_cameras.py
   ```

## Usage

### Starting the Application

1. **Run the main application**:
   
   **For MediaPipe version** (recommended if available):
   ```bash
   python src/main.py
   ```
   
   **For OpenCV-only version** (if MediaPipe is not available):
   ```bash
   python src/main_opencv.py
   ```
   
   **To use a specific camera** (when multiple cameras are connected):
   ```bash
   python src/main_opencv.py 1    # Use camera index 1
   python src/main_opencv.py 2    # Use camera index 2
   ```

2. **Using the Dashboard**:
   - Click **"Start Session"** to begin monitoring
   - The live video feed will show detected faces with bounding boxes
   - Engagement metrics update in real-time on the right panel
   - The trend chart shows engagement over time
   - Click **"Stop Session"** to end monitoring
   - Use **"Export Data"** to save session results

### Understanding the Metrics

#### Overall Engagement Score (0-100%)
- **High (70-100%)**: Audience is highly engaged
- **Medium (40-70%)**: Moderate engagement levels
- **Low (0-40%)**: Low engagement, may need attention

#### Individual Metrics
- **Eye Contact**: Percentage of people looking toward camera
- **Alertness**: Average alertness level (eyes open, attentive)
- **Smiles**: Total number of smiles detected during session
- **Laughs**: Total number of laughs detected during session

### Session Data

All session data is automatically saved to `data/engagement_data.db`. Exported data includes:
- **Session Summary**: Overall statistics and duration
- **Engagement Metrics**: Time-series data of all metrics
- **Face Tracking**: Individual face data and movements
- **Trend Charts**: Visual representation of engagement over time
- **Summary Report**: Detailed analysis and insights

## Project Structure

```
AI-Audience-Engagement-Scanner/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── face_detector.py        # Face detection and tracking
│   ├── engagement_analyzer.py  # Engagement metrics calculation
│   ├── gui_dashboard.py        # Real-time GUI dashboard
│   ├── data_manager.py         # Database and export functions
│   └── utils.py               # Helper functions
├── data/                      # Session data and exports
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technical Details

### Face Detection
- Uses MediaPipe Face Detection for robust multi-face tracking
- Supports up to 50 simultaneous faces
- Optimized for audience scenarios with long-range detection

### Engagement Analysis
- **Eye Aspect Ratio (EAR)**: Measures eye openness for alertness
- **Facial Landmarks**: 468-point face mesh for detailed analysis
- **Head Pose Estimation**: Determines gaze direction
- **Expression Analysis**: Detects smiles and laughs through mouth geometry

### Performance Optimization
- Multi-threaded processing for smooth real-time operation
- Adaptive frame processing (10 FPS analysis, 30 FPS display)
- Efficient face tracking to maintain identity across frames
- Memory management for long sessions

## Configuration

### Camera Settings
The application automatically configures camera settings for optimal performance:
- Resolution: 1280x720 (720p)
- Frame Rate: 30 FPS
- Auto-exposure and white balance

### Engagement Thresholds
Default thresholds can be modified in `engagement_analyzer.py`:
- Eye Aspect Ratio threshold: 0.25
- Smile detection threshold: 0.02
- Laugh detection threshold: 0.04
- Eye contact angle threshold: 15 degrees

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Ensure camera is connected and not used by other applications
   - Try different camera indices (0, 1, 2...)
   - Check camera permissions
   - Use `python detect_cameras.py` to see available cameras

2. **Poor face detection**:
   - Ensure adequate lighting
   - Position camera to capture faces clearly
   - Avoid backlighting

3. **Low performance**:
   - Close other applications to free up CPU/memory
   - Reduce number of faces in view if possible
   - Lower camera resolution in code if needed

4. **Installation issues**:
   - Ensure Python 3.8+ is installed
   - Use virtual environment to avoid conflicts
   - Install Visual Studio Build Tools on Windows if needed

### Performance Tips
- Position camera for optimal face visibility
- Ensure consistent lighting conditions
- Close unnecessary applications during sessions
- Use SSD storage for better database performance

## Data Privacy

- All processing is done locally on your machine
- No data is sent to external servers
- Camera feed is not recorded, only metrics are saved
- Session data is stored locally in SQLite database

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for excellent face detection models
- OpenCV community for computer vision tools
- TensorFlow team for machine learning framework

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed description and system information

---

**Note**: This application is designed for ethical use in educational and professional settings. Always ensure you have appropriate consent when monitoring audience engagement.
