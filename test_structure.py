#!/usr/bin/env python3
"""
Test script to verify project structure and basic functionality
without requiring all dependencies to be installed.
"""

import os
import sys

def test_project_structure():
    """Test that all required files exist."""
    print("Testing project structure...")
    
    required_files = [
        'src/main.py',
        'src/face_detector.py',
        'src/engagement_analyzer.py',
        'src/gui_dashboard.py',
        'src/data_manager.py',
        'src/utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("\n✅ All required files present!")
    return True

def test_data_directories():
    """Test that data directories exist."""
    print("\nTesting data directories...")
    
    required_dirs = ['data', 'data/exports']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} - creating...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ {dir_path} - created")
    
    print("\n✅ All data directories ready!")
    return True

def test_python_syntax():
    """Test Python syntax of all source files."""
    print("\nTesting Python syntax...")
    
    src_files = [
        'src/main.py',
        'src/face_detector.py',
        'src/engagement_analyzer.py',
        'src/gui_dashboard.py',
        'src/data_manager.py',
        'src/utils.py'
    ]
    
    for file_path in src_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {file_path} - error: {e}")
            return False
    
    print("\n✅ All Python files have valid syntax!")
    return True

def test_requirements():
    """Test requirements.txt format."""
    print("\nTesting requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)
        
        print(f"Found {len(packages)} required packages:")
        for package in packages:
            print(f"  - {package}")
        
        print("\n✅ Requirements file is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def main():
    """Run all tests."""
    print("AI Audience Engagement Scanner - Project Structure Test")
    print("=" * 55)
    
    tests = [
        test_project_structure,
        test_data_directories,
        test_python_syntax,
        test_requirements
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the application: python3 src/main.py")
        print("3. Ensure your camera is connected and accessible")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
