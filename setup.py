pip install -r requirements.txtpip install -r requirements.txt#!/usr/bin/env python3
# Install required packages from the requirements.txt file
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

"""
Setup script for AI Audience Engagement Scanner
Handles installation and initial setup.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_camera_availability():
    """Check if camera is available."""
    print("\nChecking camera availability...")
    
    try:
        # Try to import cv2 and test camera
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("✅ Camera is available and working")
                return True
            else:
                print("❌ Camera detected but cannot capture frames")
                return False
        else:
            print("❌ No camera detected")
            return False
            
    except ImportError:
        print("⚠️  OpenCV not installed yet - camera check will be performed after installation")
        return True
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def create_virtual_environment():
    """Create and setup virtual environment."""
    print("\nSetting up virtual environment...")
    
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print("✅ Virtual environment created")
        
        # Provide activation instructions
        system = platform.system().lower()
        if system == "windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"To activate: {activate_cmd}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def setup_directories():
    """Setup required directories."""
    print("\nSetting up directories...")
    
    directories = ["data", "data/exports"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}")
        else:
            print(f"✅ {directory} already exists")
    
    return True

def run_tests():
    """Run project structure tests."""
    print("\nRunning project tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_structure.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo run the AI Audience Engagement Scanner:")
    print("1. Ensure your camera is connected")
    print("2. Run: python3 src/main.py")
    print("\nFeatures:")
    print("• Real-time face detection and tracking")
    print("• Eye contact and alertness monitoring")
    print("• Smile and laugh detection")
    print("• Engagement trend analysis")
    print("• Data export and reporting")
    print("\nFor help and troubleshooting, see README.md")
    print("=" * 60)

def main():
    """Main setup function."""
    print("AI Audience Engagement Scanner - Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        sys.exit(1)
    
    # Ask user about virtual environment
    use_venv = input("\nCreate virtual environment? (recommended) [y/N]: ").lower().strip()
    if use_venv in ['y', 'yes']:
        if not create_virtual_environment():
            print("⚠️  Continuing without virtual environment...")
    
    # Install dependencies
    install_deps = input("\nInstall dependencies now? [Y/n]: ").lower().strip()
    if install_deps not in ['n', 'no']:
        if not install_dependencies():
            print("❌ Setup incomplete - please install dependencies manually")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)
        
        # Check camera after installation
        check_camera_availability()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed - please check the issues above")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
