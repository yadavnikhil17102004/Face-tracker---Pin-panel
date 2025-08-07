import cv2
import time
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
from datetime import datetime
import pygame
from queue import Queue
import numpy as np
import json
import csv
from collections import deque
import logging

"""
ATM Security Prototype

This application demonstrates an ATM security system with:
- Real-time face detection and tracking
- Virtual keypad window for PIN entry simulation
- Warning system when multiple people are detected
- Audio alerts for security breaches
- Screenshot capture for security incidents
"""

# Configuration parameters
CONFIG = {
    # Camera settings
    'camera_id': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'flip_horizontal': True,
    
    # Face detection settings
    'face_box_color': (0, 255, 0),  # Green for single person
    'warning_box_color': (0, 0, 255),  # Red for multiple people
    'face_box_thickness': 3,
    'min_face_size': (60, 60),  # Increased minimum face size
    'scale_factor': 1.2,        # Less sensitive scale factor
    'min_neighbors': 7,         # Increased min neighbors for more reliability
    
    # DNN-based detection
    'use_dnn_detector': True,   # Use DNN-based detection instead of Haar cascade
    'dnn_confidence': 0.5,      # Minimum confidence for DNN detections (lowered for better detection)
    'dnn_model_type': 'caffe',  # Use Caffe SSD model (better than TensorFlow)
    'caffe_model_file': 'Testing--res10_300x300_ssd_iter_140000/res10_300x300_ssd_iter_140000.caffemodel',
    'caffe_config_file': 'Testing--res10_300x300_ssd_iter_140000/deploy.prototxt',
    
    # Face filtering parameters
    'min_face_area': 0.01,      # Minimum face area as fraction of frame
    'max_face_area': 0.4,       # Maximum face area as fraction of frame
    'detection_persistence': 3, # Number of frames a face must be detected to be counted
    
    # Kalman filter parameters
    'use_kalman_filter': True,  # Use Kalman filter for face tracking
    'process_noise': 0.03,      # Process noise for Kalman filter
    'measurement_noise': 0.1,   # Measurement noise for Kalman filter
    
    # Anti-spoofing
    'enable_anti_spoofing': False,  # Enable anti-spoofing detection
    'blink_detection': True,       # Enable blink detection as part of anti-spoofing
    'texture_analysis': True,      # Enable texture analysis for anti-spoofing
    
    # Logging and analytics
    'enable_logging': True,        # Enable detailed event logging
    'log_dir': 'logs',             # Directory for log files
    'analytics_interval': 3600,    # Generate analytics every N seconds (3600 = 1 hour)
    
    # Security settings
    'max_safe_people': 1,  # Maximum safe number of people
    'warning_duration': 3,  # Duration to show warning (seconds)
    'auto_screenshot': True,  # Auto screenshot on security breach
    
    # Display settings
    'show_fps': True,
    'text_color': (255, 255, 255),  # White text
    'warning_text_color': (0, 0, 255),  # Red warning text
    'text_size': 0.8,
    'text_thickness': 2,
    
    # Screenshot settings
    'screenshot_dir': 'security_screenshots',
    'screenshot_format': 'jpg',
}

class SecurityMonitor:
    """Background thread that handles camera processing"""
    def __init__(self, event_queue):
        self.event_queue = event_queue
        self.running = False
        self.thread = None
        
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Set up logging
        if CONFIG['enable_logging']:
            self.setup_logging()
        
        # Initialize face detection models
        self.initialize_face_detection()
        
        # Security state variables
        self.security_breach = False
        self.warning_start_time = 0
        self.breach_count = 0
        
        # Face tracking for improved reliability
        self.face_history = []  # Track faces across frames
        self.face_detection_count = {}  # Count consecutive detections
        self.tracked_faces = []  # Currently tracked faces with their metadata
        self.next_face_id = 1  # Unique ID for each tracked face
        
        # Kalman filter variables
        if CONFIG['use_kalman_filter']:
            self.kalman_filters = {}  # Dict of Kalman filters for each face
        
        # Anti-spoofing variables
        if CONFIG['enable_anti_spoofing']:
            self.blink_counters = {}  # Track blinks for liveness detection
            self.texture_history = {}  # Track texture variance for anti-spoofing
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Analytics data
        self.face_count_history = deque(maxlen=100)  # Store recent face counts
        self.breach_timestamps = []  # Store timestamps of breaches
        self.last_analytics_time = time.time()
        
        # Initialize camera
        self.cap = None
        
        # Create warning sounds
        self.create_warning_sounds()
        
        # Ensure directories exist
        self.ensure_dir(CONFIG['screenshot_dir'])
        if CONFIG['enable_logging']:
            self.ensure_dir(CONFIG['log_dir'])
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.show_help = True
    
    def setup_logging(self):
        """Set up logging for security events."""
        # Make sure the log directory exists
        self.ensure_dir(CONFIG['log_dir'])
        
        log_file = os.path.join(CONFIG['log_dir'], f"security_log_{datetime.now().strftime('%Y%m%d')}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('ATM_Security')
        self.logger.info("ATM Security System initialized")
    
    def initialize_face_detection(self):
        """Initialize face detection models."""
        # Always load the Haar cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize DNN-based face detector if enabled
        if CONFIG['use_dnn_detector']:
            try:
                # Check if we should use the superior Caffe SSD model
                if CONFIG.get('dnn_model_type') == 'caffe':
                    # Use the superior Caffe SSD ResNet-10 model
                    caffe_model = CONFIG.get('caffe_model_file', 'Testing--res10_300x300_ssd_iter_140000/res10_300x300_ssd_iter_140000.caffemodel')
                    caffe_config = CONFIG.get('caffe_config_file', 'Testing--res10_300x300_ssd_iter_140000/deploy.prototxt')
                    
                    if os.path.exists(caffe_model) and os.path.exists(caffe_config):
                        try:
                            self.dnn_face_detector = cv2.dnn.readNetFromCaffe(caffe_config, caffe_model)
                            self.dnn_available = True
                            self.dnn_model_type = 'caffe'
                            print(f"Caffe SSD face detector initialized successfully using {caffe_model}")
                            if CONFIG['enable_logging']:
                                self.log_event("system", f"Caffe SSD face detector initialized with {caffe_model}")
                        except Exception as e:
                            print(f"Error loading Caffe model: {e}")
                            self.dnn_available = False
                    else:
                        print(f"Caffe model files not found at {caffe_model} or {caffe_config}")
                        print("Falling back to TensorFlow model search...")
                        self.dnn_available = False
                else:
                    # Fallback to TensorFlow model search
                    self.dnn_available = False
                
                # If Caffe model failed, try TensorFlow models as fallback
                if not self.dnn_available:
                    self.ensure_dir("models")
                    
                    # Check for TensorFlow model files in different possible locations
                    tf_paths = [
                        ("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt"),
                        (os.path.join("models", "opencv_face_detector_uint8.pb"), 
                         os.path.join("models", "opencv_face_detector.pbtxt")),
                        ("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
                    ]
                    
                    for model_file, config_file in tf_paths:
                        if os.path.exists(model_file) and os.path.exists(config_file):
                            try:
                                self.dnn_face_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                                self.dnn_available = True
                                self.dnn_model_type = 'tensorflow'
                                print(f"TensorFlow face detector initialized successfully using {model_file}")
                                if CONFIG['enable_logging']:
                                    self.log_event("system", f"TensorFlow face detector initialized with {model_file}")
                                break
                            except Exception as e:
                                print(f"Error loading TensorFlow model from {model_file}: {e}")
                    
                    if not self.dnn_available:
                        print("No DNN model files found. Using Haar cascade instead.")
                        print("For best results, ensure Caffe model files are available:")
                        print(f"  - {CONFIG.get('caffe_model_file')}")
                        print(f"  - {CONFIG.get('caffe_config_file')}")
                        
            except Exception as e:
                print(f"Error initializing DNN face detector: {e}")
                self.dnn_available = False
                if CONFIG['enable_logging']:
                    self.log_event("error", f"DNN initialization failed: {e}")
        else:
            self.dnn_available = False
    
    def create_warning_sounds(self):
        """Create different warning sounds for various security events."""
        try:
            # Create a simple beep sound using pygame
            sample_rate = 22050
            
            # Create breach warning sound (higher pitch)
            duration = 0.5
            frequency = 800
            buffer = np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate).astype(np.float32)
            buffer = (buffer * 32767).astype(np.int16)
            stereo_buffer = np.column_stack((buffer, buffer))
            self.warning_sound = pygame.mixer.Sound(stereo_buffer)
            
            # Create anti-spoofing warning sound (lower pitch, longer)
            duration = 0.7
            frequency = 500
            buffer = np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate).astype(np.float32)
            buffer = (buffer * 32767).astype(np.int16)
            stereo_buffer = np.column_stack((buffer, buffer))
            self.spoof_warning_sound = pygame.mixer.Sound(stereo_buffer)
            
        except Exception as e:
            print(f"Warning: Could not create warning sounds: {e}")
            self.warning_sound = None
            self.spoof_warning_sound = None
    
    def create_kalman_filter(self):
        """Create a Kalman filter for face tracking."""
        # State: [x, y, width, height, dx, dy, dw, dh]
        # where dx, dy, dw, dh are the velocities
        kalman = cv2.KalmanFilter(8, 4)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Process noise
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * CONFIG['process_noise']
        
        # Measurement noise
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * CONFIG['measurement_noise']
        
        return kalman
    
    def detect_faces_dnn(self, frame, return_confidence=False):
        """Detect faces using DNN-based detector (Caffe SSD or TensorFlow)."""
        if not self.dnn_available:
            return None
        
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Handle different model types
            if hasattr(self, 'dnn_model_type') and self.dnn_model_type == 'caffe':
                # Use Caffe SSD model (superior detection)
                # Resize and create blob with mean subtraction values specific to this model
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)  # Mean subtraction values for SSD
                )
                self.dnn_face_detector.setInput(blob)
                detections = self.dnn_face_detector.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > CONFIG['dnn_confidence']:
                        # Get bounding box coordinates
                        box = detections[0, 0, i, 3:7] * [frame_width, frame_height, frame_width, frame_height]
                        (x1, y1, x2, y2) = box.astype("int")
                        
                        # Ensure coordinates are within frame boundaries
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width - 1))
                        y2 = max(0, min(y2, frame_height - 1))
                        
                        # Only add valid detections
                        if x2 > x1 and y2 > y1:
                            # Convert to same format as Haar cascade (x, y, w, h)
                            if return_confidence:
                                faces.append((x1, y1, x2 - x1, y2 - y1, confidence * 100))
                            else:
                                faces.append((x1, y1, x2 - x1, y2 - y1))
                
            else:
                # Use TensorFlow model (fallback)
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                self.dnn_face_detector.setInput(blob)
                detections = self.dnn_face_detector.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > CONFIG['dnn_confidence']:
                        x1 = int(detections[0, 0, i, 3] * frame_width)
                        y1 = int(detections[0, 0, i, 4] * frame_height)
                        x2 = int(detections[0, 0, i, 5] * frame_width)
                        y2 = int(detections[0, 0, i, 6] * frame_height)
                        
                        # Ensure coordinates are within frame boundaries
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width - 1))
                        y2 = max(0, min(y2, frame_height - 1))
                        
                        # Only add valid detections
                        if x2 > x1 and y2 > y1:
                            # Convert to same format as Haar cascade (x, y, w, h)
                            if return_confidence:
                                faces.append((x1, y1, x2 - x1, y2 - y1, confidence * 100))
                            else:
                                faces.append((x1, y1, x2 - x1, y2 - y1))
            
            return faces
            
        except Exception as e:
            print(f"Error in DNN face detection: {e}")
            if CONFIG['enable_logging']:
                self.log_event("error", f"DNN face detection error: {e}")
            self.dnn_available = False  # Disable DNN after error
            return None
    
    def check_anti_spoofing(self, frame, face_rect):
        """Check if a face is real or a spoof (photo/screen)."""
        x, y, w, h = face_rect
        face_id = f"{x//20}_{y//20}_{w//20}_{h//20}"
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        if face_region.size == 0:  # Skip if face region is invalid
            return True  # Assume real for safety
        
        is_real = True
        reasons = []
        
        # Check 1: Blink detection (if enabled)
        if CONFIG['blink_detection']:
            is_blinking, blink_count = self.detect_blink(face_region, face_id)
            if face_id not in self.blink_counters:
                self.blink_counters[face_id] = {'count': 0, 'frames': 0, 'last_blink': False}
            
            # Update blink counter
            self.blink_counters[face_id]['frames'] += 1
            if is_blinking and not self.blink_counters[face_id]['last_blink']:
                self.blink_counters[face_id]['count'] += 1
            self.blink_counters[face_id]['last_blink'] = is_blinking
            
            # Check if blinking rate is too low (possible spoof)
            if self.blink_counters[face_id]['frames'] > 90:  # After 3 seconds (at 30fps)
                if self.blink_counters[face_id]['count'] == 0:
                    is_real = False
                    reasons.append("No blinks detected")
        
        # Check 2: Texture analysis (if enabled)
        if CONFIG['texture_analysis']:
            # Calculate texture variance using Laplacian
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            blur_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Track texture history
            if face_id not in self.texture_history:
                self.texture_history[face_id] = []
            
            self.texture_history[face_id].append(blur_measure)
            if len(self.texture_history[face_id]) > 10:
                self.texture_history[face_id].pop(0)
            
            # Check texture variance (printed photos have lower variance)
            if len(self.texture_history[face_id]) >= 5:
                avg_variance = sum(self.texture_history[face_id]) / len(self.texture_history[face_id])
                if avg_variance < 50:  # Threshold for printed photo
                    is_real = False
                    reasons.append("Low texture variance")
        
        # Log spoofing attempts
        if not is_real and CONFIG['enable_logging']:
            reason_str = ", ".join(reasons)
            self.logger.warning(f"Possible spoofing attempt detected: {reason_str}")
        
        return is_real
    
    def detect_blink(self, face_region, face_id):
        """Detect blinking for liveness detection."""
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Simple blink detection based on eye detection
        # When eyes close, they may not be detected
        is_blinking = len(eyes) < 2
        return is_blinking, len(eyes)
    
    def generate_analytics(self):
        """Generate analytics from security data."""
        if not CONFIG['enable_logging']:
            return
        
        current_time = time.time()
        
        # Only generate analytics at the specified interval
        if current_time - self.last_analytics_time < CONFIG['analytics_interval']:
            return
        
        self.last_analytics_time = current_time
        
        # Calculate statistics
        avg_face_count = sum(self.face_count_history) / len(self.face_count_history) if self.face_count_history else 0
        max_face_count = max(self.face_count_history) if self.face_count_history else 0
        
        # Count breaches in the last hour
        recent_breaches = 0
        hour_ago = current_time - 3600
        for timestamp in self.breach_timestamps:
            if timestamp > hour_ago:
                recent_breaches += 1
        
        # Log analytics
        self.logger.info(f"ANALYTICS: Avg faces: {avg_face_count:.1f}, Max faces: {max_face_count}, Recent breaches: {recent_breaches}")
        
        # Save analytics to CSV
        self.save_analytics_to_csv(avg_face_count, max_face_count, recent_breaches)
    
    def save_analytics_to_csv(self, avg_face_count, max_face_count, recent_breaches):
        """Save analytics data to CSV file."""
        csv_file = os.path.join(CONFIG['log_dir'], f"security_analytics_{datetime.now().strftime('%Y%m%d')}.csv")
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Avg Faces', 'Max Faces', 'Recent Breaches', 'Total Breaches'])
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{avg_face_count:.1f}",
                max_face_count,
                recent_breaches,
                self.breach_count
            ])
    
    def log_event(self, event_type, details=None):
        """Log a security event."""
        if not CONFIG['enable_logging'] or not hasattr(self, 'logger'):
            return
        
        if event_type == "breach":
            self.logger.warning(f"Security breach: {details}")
            self.breach_timestamps.append(time.time())
        elif event_type == "normal":
            self.logger.info(f"Security returned to normal: {details}")
        elif event_type == "spoof":
            self.logger.warning(f"Spoofing attempt detected: {details}")
        elif event_type == "system":
            self.logger.info(f"System event: {details}")
        elif event_type == "error":
            self.logger.error(f"Error: {details}")
        else:
            self.logger.info(f"{event_type}: {details}")
    
    def ensure_dir(self, directory):
        """Ensure that a directory exists, creating it if necessary."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            if CONFIG['enable_logging'] and hasattr(self, 'logger'):
                self.logger.info(f"Created directory: {directory}")
    
    def check_security_breach(self, face_count, spoofing_detected=False):
        """Check for security breaches and update system state."""
        # Determine if there's a security breach
        is_breach = face_count > CONFIG['max_safe_people'] or spoofing_detected
        
        # If status changed, take appropriate actions
        if is_breach and not self.security_breach:
            # New security breach
            self.security_breach = True
            self.warning_start_time = time.time()
            self.breach_count += 1
            
            # Log the breach
            breach_type = "Multiple people detected" if face_count > CONFIG['max_safe_people'] else "Spoofing attempt"
            if CONFIG['enable_logging']:
                self.log_event("breach", f"{breach_type} - Face count: {face_count}")
            
            # Play warning sound
            if spoofing_detected and hasattr(self, 'spoof_warning_sound') and self.spoof_warning_sound:
                self.spoof_warning_sound.play()
            elif hasattr(self, 'warning_sound') and self.warning_sound:
                self.warning_sound.play()
            
            # Notify the main app thread
            self.event_queue.put(("security_breach", face_count))
            
        elif not is_breach and self.security_breach:
            # Security breach ended
            self.security_breach = False
            
            # Log return to normal
            if CONFIG['enable_logging']:
                self.log_event("normal", f"Face count: {face_count}")
            
            # Notify the main app thread
            self.event_queue.put(("security_normal", face_count))
            
        # Continue to update state if breach is ongoing
        elif is_breach and self.security_breach:
            # If warning duration has passed, reset for new warnings
            if time.time() - self.warning_start_time > CONFIG['warning_duration']:
                # Reset warning
                self.warning_start_time = time.time()
                
                # Play warning sound at intervals
                if spoofing_detected and hasattr(self, 'spoof_warning_sound') and self.spoof_warning_sound:
                    self.spoof_warning_sound.play()
                elif hasattr(self, 'warning_sound') and self.warning_sound:
                    self.warning_sound.play()
    
    def take_screenshot(self, frame, reason="security_breach"):
        """Save a screenshot for security purposes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CONFIG['screenshot_dir']}/{reason}_{timestamp}.{CONFIG['screenshot_format']}"
        cv2.imwrite(filename, frame)
        
        if CONFIG['enable_logging'] and hasattr(self, 'logger'):
            self.logger.info(f"Screenshot saved: {filename}")
        
        return filename
    
    def detect_faces_with_filtering(self, frame):
        """Detect faces with improved reliability through filtering and tracking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mild blur to reduce noise (helps with false detections)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Improve contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect potential faces
        potential_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces by size and position to reduce false positives
        filtered_faces = []
        frame_height, frame_width = frame.shape[:2]
        min_face_area = CONFIG.get('min_face_area', 0.01) * frame_width * frame_height
        max_face_area = CONFIG.get('max_face_area', 0.4) * frame_width * frame_height
        
        for (x, y, w, h) in potential_faces:
            face_area = w * h
            
            # Size-based filtering
            if face_area < min_face_area or face_area > max_face_area:
                continue
            
            # Position filtering - eliminate faces that are unlikely (like edges)
            if y < 0.05 * frame_height or (y + h) > 0.95 * frame_height:
                continue
                
            # Additional criteria - width to height ratio should be reasonable for a face
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue
            
            filtered_faces.append((x, y, w, h))
        
        # Use face detection persistence to reduce false positives
        persistence_threshold = CONFIG.get('detection_persistence', 3)
        
        # Update face detection count
        current_face_ids = {}
        
        for face_rect in filtered_faces:
            x, y, w, h = face_rect
            face_id = f"{x//20}_{y//20}_{w//20}_{h//20}"  # Create a rough face ID based on position
            current_face_ids[face_id] = face_rect
            
            if face_id in self.face_detection_count:
                self.face_detection_count[face_id] += 1
            else:
                self.face_detection_count[face_id] = 1
        
        # Remove faces that are no longer detected
        for face_id in list(self.face_detection_count.keys()):
            if face_id not in current_face_ids:
                self.face_detection_count[face_id] -= 1
                if self.face_detection_count[face_id] <= 0:
                    del self.face_detection_count[face_id]
        
        # Only count faces that have been detected consistently
        consistent_faces = []
        for face_id, count in self.face_detection_count.items():
            if count >= persistence_threshold and face_id in current_face_ids:
                consistent_faces.append(current_face_ids[face_id])
        
        return consistent_faces
    def detect_faces_with_filtering(self, frame):
        """Detect faces with improved reliability through filtering and tracking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mild blur to reduce noise (helps with false detections)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Improve contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect potential faces
        potential_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces by size and position to reduce false positives
        filtered_faces = []
        frame_height, frame_width = frame.shape[:2]
        min_face_area = CONFIG.get('min_face_area', 0.01) * frame_width * frame_height
        max_face_area = CONFIG.get('max_face_area', 0.4) * frame_width * frame_height
        
        for (x, y, w, h) in potential_faces:
            face_area = w * h
            
            # Size-based filtering
            if face_area < min_face_area or face_area > max_face_area:
                continue
            
            # Position filtering - eliminate faces that are unlikely (like edges)
            if y < 0.05 * frame_height or (y + h) > 0.95 * frame_height:
                continue
                
            # Additional criteria - width to height ratio should be reasonable for a face
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue
            
            filtered_faces.append((x, y, w, h))
        
        # Use face detection persistence to reduce false positives
        persistence_threshold = CONFIG.get('detection_persistence', 3)
        
        # Update face detection count
        current_face_ids = {}
        
        for face_rect in filtered_faces:
            x, y, w, h = face_rect
            face_id = f"{x//20}_{y//20}_{w//20}_{h//20}"  # Create a rough face ID based on position
            current_face_ids[face_id] = face_rect
            
            if face_id in self.face_detection_count:
                self.face_detection_count[face_id] += 1
            else:
                self.face_detection_count[face_id] = 1
        
        # Remove faces that are no longer detected
        for face_id in list(self.face_detection_count.keys()):
            if face_id not in current_face_ids:
                self.face_detection_count[face_id] -= 1
                if self.face_detection_count[face_id] <= 0:
                    del self.face_detection_count[face_id]
        
        # Only count faces that have been detected consistently
        consistent_faces = []
        for face_id, count in self.face_detection_count.items():
            if count >= persistence_threshold and face_id in current_face_ids:
                consistent_faces.append(current_face_ids[face_id])
        
        return consistent_faces
    
    def start(self):
        """Start the security monitor thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Security monitor is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print("Security monitor started")
    
    def stop(self):
        """Stop the security monitor thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Security monitor stopped")
    
    def run(self):
        """Main processing loop."""
        print("🏧 ATM Security System Started")
        print("Features:")
        print("  - Superior SSD ResNet-10 face detection (if available)")
        print("  - Deep learning-based face detection with confidence scores")
        print("  - Kalman filter face tracking")
        print("  - Anti-spoofing detection (disabled)")
        print("  - Security event logging and analytics")
        print("  - Multiple person warning system")
        print("  - Automatic security screenshots")
        print("  - Virtual keypad simulation")
        print("\nControls:")
        print("  'q' - Quit application")
        print("  's' - Manual screenshot")
        print("  'k' - Open/Close keypad window")
        print("  'h' - Toggle help display")
        print("  'a' - Show analytics report")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CONFIG['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            if CONFIG['enable_logging']:
                self.log_event("error", "Could not open camera")
            self.event_queue.put(("error", "Could not open camera"))
            return
        
        # Reset FPS counter
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize session
        if CONFIG['enable_logging']:
            self.log_event("system", "Security monitoring session started")
        
        # Spoofing detection variables
        spoofing_detected = False
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                if CONFIG['enable_logging']:
                    self.log_event("error", "Failed to capture frame")
                break
            
            if CONFIG['flip_horizontal']:
                frame = cv2.flip(frame, 1)
            
            # Detect faces with improved reliability using superior SSD model
            detected_faces = []
            detection_method = "Unknown"
            
            if CONFIG['use_dnn_detector'] and self.dnn_available:
                # Try DNN detection first (much more reliable) with confidence values
                dnn_faces = self.detect_faces_dnn(frame, return_confidence=True)
                if dnn_faces is not None:
                    detected_faces = dnn_faces
                    detection_method = f"DNN-{getattr(self, 'dnn_model_type', 'Unknown').upper()}"
                    # Don't fall back to Haar cascade if DNN is working
                else:
                    # Only fall back to Haar cascade if DNN completely fails
                    detected_faces = self.detect_faces_with_filtering(frame)
                    detection_method = "Haar Cascade (DNN Failed)"
            else:
                # Use Haar cascade if DNN is not available
                detected_faces = self.detect_faces_with_filtering(frame)
                detection_method = "Haar Cascade"
            
            # Check for spoofing if enabled
            spoofing_detected = False
            if CONFIG['enable_anti_spoofing'] and len(detected_faces) > 0:
                for face_rect in detected_faces:
                    if not self.check_anti_spoofing(frame, face_rect):
                        spoofing_detected = True
                        cv2.putText(frame, "SPOOF DETECTED", 
                                  (face_rect[0], face_rect[1] - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        break
            
            face_count = len(detected_faces)
            
            # Store face count for analytics
            if CONFIG['enable_logging']:
                self.face_count_history.append(face_count)
            
            # Update UI about face count
            self.event_queue.put(("face_count", face_count))
            
            # Check for security breach
            self.check_security_breach(face_count, spoofing_detected)
            
            # Generate analytics periodically
            if CONFIG['enable_logging']:
                self.generate_analytics()
            
            # Choose box color based on security status
            box_color = CONFIG['warning_box_color'] if self.security_breach else CONFIG['face_box_color']
            
            # Draw face rectangles with enhanced information
            for i, face_data in enumerate(detected_faces):
                if isinstance(face_data, tuple) and len(face_data) == 4:
                    # Standard format (x, y, w, h)
                    x, y, w, h = face_data
                    confidence = None
                elif isinstance(face_data, tuple) and len(face_data) == 5:
                    # Enhanced format with confidence (x, y, w, h, confidence)
                    x, y, w, h, confidence = face_data
                else:
                    continue
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, CONFIG['face_box_thickness'])
                
                # Add face number and confidence if available
                person_index = i + 1
                if confidence is not None:
                    label = f"Person {person_index} ({confidence:.1f}%)"
                else:
                    label = f"Person {person_index}"
                
                cv2.putText(frame, label, 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # If using Kalman tracking, draw predicted trajectory
                if CONFIG['use_kalman_filter'] and hasattr(self, 'kalman_filters'):
                    for face_id, kalman in self.kalman_filters.items():
                        # Get state prediction for 5 frames ahead
                        predicted_state = kalman.statePost.copy()
                        future_positions = []
                        for j in range(5):
                            # Apply transition matrix manually
                            predicted_state[0] += predicted_state[4]  # x += dx
                            predicted_state[1] += predicted_state[5]  # y += dy
                            predicted_state[2] += predicted_state[6]  # w += dw
                            predicted_state[3] += predicted_state[7]  # h += dh
                            
                            # Extract position
                            pred_x = int(predicted_state[0])
                            pred_y = int(predicted_state[1])
                            future_positions.append((pred_x, pred_y))
                        
                        # Draw trajectory
                        for j in range(1, len(future_positions)):
                            cv2.line(frame, future_positions[j-1], future_positions[j], (0, 255, 255), 1)
            
            # Auto screenshot on security breach
            if self.security_breach and CONFIG['auto_screenshot']:
                if time.time() - self.warning_start_time < 0.5:  # Screenshot once per breach
                    filename = self.take_screenshot(frame)
                    print(f"🔴 Security screenshot saved: {filename}")
                    self.event_queue.put(("screenshot", filename))
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Display information
            y_pos = 30
            
            # Security status
            if self.security_breach:
                status_text = f"⚠️  SECURITY BREACH - {face_count} PEOPLE DETECTED! ⚠️"
                if spoofing_detected:
                    status_text = "⚠️  SECURITY BREACH - SPOOFING ATTEMPT DETECTED! ⚠️"
                
                cv2.putText(frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, CONFIG['warning_text_color'], 2)
                y_pos += 30
                
                # Warning timer
                remaining_time = CONFIG['warning_duration'] - (time.time() - self.warning_start_time)
                if remaining_time > 0:
                    timer_text = f"Warning active: {remaining_time:.1f}s"
                    cv2.putText(frame, timer_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, CONFIG['warning_text_color'], 2)
                    y_pos += 25
            else:
                status_text = f"✅ SECURE - {face_count} person(s) detected"
                cv2.putText(frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, CONFIG['text_color'], 2)
                y_pos += 30
            
            # Display detection method with more detail
            if detection_method.startswith("DNN-CAFFE"):
                cv2.putText(frame, "SSD ResNet-10 Detection (Superior)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)  # Green for superior detection
            elif detection_method.startswith("DNN-TENSORFLOW"):
                cv2.putText(frame, "TensorFlow DNN Detection", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 0), 2)  # Yellow for good detection
            elif "Haar Cascade" in detection_method:
                cv2.putText(frame, detection_method, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 0, 0), 2)  # Red for basic detection
            else:
                cv2.putText(frame, detection_method, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, CONFIG['text_color'], 2)
            y_pos += 25
            
            # Additional info
            if CONFIG['show_fps']:
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, CONFIG['text_color'], 2)
                y_pos += 25
            
            cv2.putText(frame, f"Security Breaches: {self.breach_count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG['text_color'], 2)
            y_pos += 25
            
            # Analytics info if enabled
            if CONFIG['enable_logging'] and self.face_count_history:
                avg_faces = sum(self.face_count_history) / len(self.face_count_history)
                cv2.putText(frame, f"Avg Faces: {avg_faces:.1f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG['text_color'], 2)
                y_pos += 25
            
            # Help display
            if self.show_help:
                help_y = CONFIG['frame_height'] - 120
                cv2.putText(frame, "Controls:", 
                           (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
                help_y += 20
                cv2.putText(frame, "q=Quit | s=Screenshot | k=Keypad | h=Help | a=Analytics", 
                           (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
                help_y += 20
                if CONFIG['enable_anti_spoofing']:
                    cv2.putText(frame, "Anti-spoofing: Active", 
                               (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
                if CONFIG['use_kalman_filter']:
                    cv2.putText(frame, "Kalman tracking: Active", 
                               (200, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
            
            # Display frame
            cv2.imshow('ATM Security Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if CONFIG['enable_logging']:
                    self.log_event("system", "User quit application")
                self.event_queue.put(("quit", None))
                break
            elif key == ord('s'):
                filename = self.take_screenshot(frame, "manual")
                print(f"📸 Manual screenshot saved: {filename}")
                if CONFIG['enable_logging']:
                    self.log_event("system", f"Manual screenshot saved: {filename}")
                self.event_queue.put(("manual_screenshot", filename))
            elif key == ord('k'):
                if CONFIG['enable_logging']:
                    self.log_event("system", "Keypad toggled")
                self.event_queue.put(("toggle_keypad", None))
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('a'):
                if CONFIG['enable_logging']:
                    # Display analytics report
                    self.display_analytics_report()
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        
        # Log session end
        if CONFIG['enable_logging']:
            self.log_event("system", f"Security monitoring session ended. Total breaches: {self.breach_count}")
            
        print(f"\n🏧 ATM Security System Stopped")
        print(f"Total security breaches detected: {self.breach_count}")
    
    def display_analytics_report(self):
        """Display a detailed analytics report window."""
        if not CONFIG['enable_logging'] or not self.face_count_history:
            print("No analytics data available")
            return
        
        # Create a blank image for the report
        report_img = np.zeros((480, 640, 3), dtype=np.uint8)
        report_img[:] = (50, 50, 50)  # Dark gray background
        
        # Title
        cv2.putText(report_img, "ATM Security Analytics Report", (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current date and time
        cv2.putText(report_img, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (20, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Statistics
        y_pos = 100
        avg_faces = sum(self.face_count_history) / len(self.face_count_history)
        max_faces = max(self.face_count_history) if self.face_count_history else 0
        
        cv2.putText(report_img, f"Session Duration: {(time.time() - self.start_time)/60:.1f} minutes", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        cv2.putText(report_img, f"Security Breaches: {self.breach_count}", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        cv2.putText(report_img, f"Average Faces Detected: {avg_faces:.2f}", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        cv2.putText(report_img, f"Maximum Faces Detected: {max_faces}", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        # Count breaches in the last hour
        recent_breaches = 0
        hour_ago = time.time() - 3600
        for timestamp in self.breach_timestamps:
            if timestamp > hour_ago:
                recent_breaches += 1
        
        cv2.putText(report_img, f"Breaches in Last Hour: {recent_breaches}", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 50
        
        # System information
        cv2.putText(report_img, "System Information:", (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        if CONFIG['use_dnn_detector'] and hasattr(self, 'dnn_available') and self.dnn_available:
            if hasattr(self, 'dnn_model_type') and self.dnn_model_type == 'caffe':
                detector_text = "Face Detection: SSD ResNet-10 (Superior)"
            elif hasattr(self, 'dnn_model_type') and self.dnn_model_type == 'tensorflow':
                detector_text = "Face Detection: TensorFlow DNN"
            else:
                detector_text = "Face Detection: DNN (Unknown Type)"
        else:
            detector_text = "Face Detection: Haar Cascade Classifier"
        cv2.putText(report_img, detector_text, (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        
        if CONFIG['use_kalman_filter']:
            cv2.putText(report_img, "Tracking: Kalman Filter", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(report_img, "Tracking: Persistence-based", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        
        if CONFIG['enable_anti_spoofing']:
            cv2.putText(report_img, "Anti-spoofing: Enabled", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(report_img, "Anti-spoofing: Disabled", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 50
        
        # Log info
        if CONFIG['enable_logging']:
            log_file = os.path.join(CONFIG['log_dir'], f"security_log_{datetime.now().strftime('%Y%m%d')}.log")
            csv_file = os.path.join(CONFIG['log_dir'], f"security_analytics_{datetime.now().strftime('%Y%m%d')}.csv")
            
            cv2.putText(report_img, "Log Files:", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y_pos += 25
            
            cv2.putText(report_img, f"Event Log: {os.path.basename(log_file)}", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            
            cv2.putText(report_img, f"Analytics: {os.path.basename(csv_file)}", (20, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Footer
        cv2.putText(report_img, "Press any key to close this report", (20, 450), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display the report
        cv2.imshow('ATM Security Analytics', report_img)
        cv2.waitKey(0)
        cv2.destroyWindow('ATM Security Analytics')


class ATMKeypadWindow:
    """Keypad UI for ATM simulation"""
    def __init__(self, root, event_queue):
        self.root = root
        self.event_queue = event_queue
        
        self.pin_entry = ""
        self.security_breach = False
        
        # Configure the main window
        self.root.title("ATM Security Prototype - PIN Entry")
        self.root.geometry("400x600")
        self.root.configure(bg='#2c3e50')
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create UI elements
        self.create_ui()
    
    def create_ui(self):
        """Create all UI elements"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🏧 ATM SECURITY PROTOTYPE", 
            font=('Arial', 16, 'bold'), 
            bg='#2c3e50', 
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Security status display
        self.security_status = tk.Label(
            self.root,
            text="🟢 SECURE - Single User Detected",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='#27ae60'
        )
        self.security_status.pack(pady=10)
        
        # PIN display
        self.pin_display = tk.Label(
            self.root,
            text="Enter PIN: ____",
            font=('Arial', 14),
            bg='#34495e',
            fg='white',
            relief='sunken',
            width=20,
            height=2
        )
        self.pin_display.pack(pady=20)
        
        # Keypad frame
        keypad_frame = tk.Frame(self.root, bg='#2c3e50')
        keypad_frame.pack(pady=20)
        
        # Create number buttons
        buttons = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['*', '0', '#']
        ]
        
        for i, row in enumerate(buttons):
            for j, num in enumerate(row):
                btn = tk.Button(
                    keypad_frame,
                    text=num,
                    font=('Arial', 16, 'bold'),
                    width=5,
                    height=2,
                    bg='#3498db',
                    fg='white',
                    command=lambda n=num: self.keypad_press(n)
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
        
        # Action buttons
        action_frame = tk.Frame(self.root, bg='#2c3e50')
        action_frame.pack(pady=20)
        
        clear_btn = tk.Button(
            action_frame,
            text="CLEAR",
            font=('Arial', 12, 'bold'),
            width=10,
            height=2,
            bg='#e74c3c',
            fg='white',
            command=self.clear_pin
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        enter_btn = tk.Button(
            action_frame,
            text="ENTER",
            font=('Arial', 12, 'bold'),
            width=10,
            height=2,
            bg='#27ae60',
            fg='white',
            command=self.enter_pin
        )
        enter_btn.pack(side=tk.LEFT, padx=10)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="This is a PROTOTYPE demonstration.\nSecurity camera monitors for multiple users.\nTransaction is blocked if breach detected.",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7',
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
    
    def update_security_status(self, status, face_count=0):
        """Update the security status display."""
        if status == "secure":
            if face_count == 0:
                self.security_status.config(
                    text="⚪ NO USERS DETECTED",
                    fg='#bdc3c7'
                )
            else:
                self.security_status.config(
                    text="🟢 SECURE - Single User Detected",
                    fg='#27ae60'
                )
            self.security_breach = False
        elif status == "breach":
            self.security_status.config(
                text=f"🔴 SECURITY BREACH - {face_count} People Detected!",
                fg='#e74c3c'
            )
            self.security_breach = True
    
    def keypad_press(self, key):
        """Handle keypad button press."""
        if self.security_breach:
            self.show_warning()
            return
        
        if key.isdigit() and len(self.pin_entry) < 4:
            self.pin_entry += key
            self.update_pin_display()
    
    def clear_pin(self):
        """Clear the PIN entry."""
        self.pin_entry = ""
        self.update_pin_display()
    
    def enter_pin(self):
        """Process PIN entry."""
        if self.security_breach:
            self.show_warning()
            return
        
        if len(self.pin_entry) == 4:
            messagebox.showinfo("Transaction", f"PIN Entered: {'*' * len(self.pin_entry)}\n\nThis is a PROTOTYPE.\nIn real ATM, transaction would proceed.")
            self.clear_pin()
        else:
            messagebox.showwarning("Invalid PIN", "Please enter a 4-digit PIN.")
    
    def update_pin_display(self):
        """Update the PIN display."""
        display_text = "Enter PIN: " + "*" * len(self.pin_entry) + "_" * (4 - len(self.pin_entry))
        self.pin_display.config(text=display_text)
    
    def show_warning(self):
        """Show security warning dialog."""
        messagebox.showwarning(
            "SECURITY ALERT", 
            "⚠️ MULTIPLE PEOPLE DETECTED! ⚠️\n\nPlease ensure you are alone\nbefore entering your PIN.\n\nTransaction is paused for security."
        )
    
    def on_close(self):
        """Handle window close event."""
        # Send close event to main app
        self.event_queue.put(("keypad_closed", None))
        self.root.withdraw()  # Hide the window instead of destroying it


class ATMSecurityApp:
    """Main application class that coordinates UI and security monitoring"""
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # Hide the main window, we'll use only the keypad
        self.root.title("ATM Security System")
        self.is_quitting = False  # Flag to prevent event processing after quit
        
        # Create event queue for thread communication
        self.event_queue = Queue()
        
        # Create security monitor
        self.security_monitor = SecurityMonitor(self.event_queue)
        
        # Create keypad window
        self.keypad_window = ATMKeypadWindow(tk.Toplevel(root), self.event_queue)
        
        # Start security monitor
        self.security_monitor.start()
        
        # Start event processing
        self.process_events()
    
    def process_events(self):
        """Process events from the security monitor thread."""
        if self.is_quitting:
            return  # Don't process events if we're quitting
            
        try:
            while not self.event_queue.empty():
                event, data = self.event_queue.get_nowait()
                
                if event == "face_count":
                    # Update security status based on face count
                    if data > CONFIG['max_safe_people']:
                        self.keypad_window.update_security_status("breach", data)
                    else:
                        self.keypad_window.update_security_status("secure", data)
                
                elif event == "security_breach":
                    # Security breach detected
                    self.keypad_window.update_security_status("breach", data)
                
                elif event == "security_normal":
                    # Security status returned to normal
                    self.keypad_window.update_security_status("secure", data)
                
                elif event == "screenshot" or event == "manual_screenshot":
                    # Screenshot taken
                    pass  # Just log it, no UI action needed
                
                elif event == "toggle_keypad":
                    # Show keypad window if it's hidden
                    self.keypad_window.root.deiconify()
                
                elif event == "keypad_closed":
                    # Keypad window was closed
                    pass  # Let it be hidden
                
                elif event == "quit":
                    # Quit the application
                    self.quit()
                    return  # Stop processing events
                
                elif event == "error":
                    # Error occurred
                    messagebox.showerror("Error", data)
        
        except Exception as e:
            print(f"Error processing events: {e}")
        
        # Schedule next event processing only if not quitting
        if not self.is_quitting:
            self.root.after(100, self.process_events)
    
    def quit(self):
        """Quit the application."""
        print("Shutting down ATM Security System...")
        
        # Set the quitting flag to stop event processing
        self.is_quitting = True
        
        # Stop the security monitor
        self.security_monitor.stop()
        
        # Destroy keypad window
        try:
            self.keypad_window.root.destroy()
        except:
            pass
        
        # Destroy main window and exit
        self.root.quit()
        self.root.destroy()


def main():
    # Create root window
    root = tk.Tk()
    
    # Create and start application
    app = ATMSecurityApp(root)
    
    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
