"""
Download script for OpenCV DNN face detection models.

This script will download the necessary model files for DNN-based face detection.
Run this script once before using the ATM Security System with DNN detection enabled.
"""

import os
import urllib.request
import sys

# URLs for the model files
MODEL_URLS = {
    "opencv_face_detector_uint8.pb": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/opencv_face_detector_uint8.pb",
    "opencv_face_detector.pbtxt": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
}

def download_models():
    """Download the model files if they don't exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Downloading DNN face detection models...")
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(script_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Skipping download.")
            continue
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename} successfully.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    print("\nAll model files downloaded successfully.")
    print("You can now use DNN-based face detection in the ATM Security System.")
    return True

if __name__ == "__main__":
    success = download_models()
    if not success:
        print("\nError: Failed to download all model files.")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    sys.exit(0)
