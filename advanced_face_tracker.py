import cv2
import time
import os
from datetime import datetime

"""
Advanced Face Tracking Application

This program extends the basic face tracker with additional features:
- Eye detection within faces
- Screenshot capture functionality
- More detailed statistics
- Additional configuration options

Configuration parameters are at the top of the file for easy customization.
"""

# Configuration parameters - modify these to customize the application
CONFIG = {
    # Camera settings
    'camera_id': 0,                # Camera ID (usually 0 for built-in webcam)
    'frame_width': 640,            # Width of the camera frame
    'frame_height': 480,           # Height of the camera frame
    'flip_horizontal': True,       # Flip the camera horizontally (mirror mode)
    
    # Face detection settings
    'face_box_color': (0, 255, 0),  # Color of face bounding box (BGR format)
    'face_box_thickness': 2,        # Thickness of bounding box lines
    'min_face_size': (30, 30),      # Minimum face size to detect
    'scale_factor': 1.1,            # How much the image size is reduced at each image scale
    'min_neighbors': 5,              # How many neighbors each candidate rectangle should have
    
    # Eye detection settings
    'detect_eyes': True,           # Whether to detect eyes within faces
    'eye_box_color': (255, 0, 0),  # Color of eye bounding box (BGR format)
    'eye_box_thickness': 1,        # Thickness of eye bounding box lines
    
    # Display settings
    'show_fps': True,              # Whether to display FPS counter
    'show_help': True,             # Whether to display help text
    'text_color': (0, 0, 255),     # Color of text overlays (BGR format)
    'text_size': 0.7,              # Size of text overlays
    'text_thickness': 2,           # Thickness of text overlays
    
    # Screenshot settings
    'screenshot_dir': 'screenshots',  # Directory to save screenshots
    'screenshot_format': 'jpg',      # Format to save screenshots (jpg or png)
}


def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def take_screenshot(frame, directory=CONFIG['screenshot_dir'], format=CONFIG['screenshot_format']):
    """Save a screenshot of the current frame."""
    ensure_dir(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/face_tracker_{timestamp}.{format}"
    cv2.imwrite(filename, frame)
    return filename


def draw_text(frame, text, position, color=CONFIG['text_color'], 
              size=CONFIG['text_size'], thickness=CONFIG['text_thickness']):
    """Draw text on the frame with consistent formatting."""
    cv2.putText(
        frame, 
        text, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        size, 
        color, 
        thickness
    )


def main():
    # Load the pre-trained face and eye detectors from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Check if the cascades loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    if CONFIG['detect_eyes'] and eye_cascade.empty():
        print("Warning: Could not load eye cascade classifier, eye detection disabled")
        CONFIG['detect_eyes'] = False
    
    # Initialize the webcam
    cap = cv2.VideoCapture(CONFIG['camera_id'])
    
    # Set the frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Advanced Face Tracker started.")
    print("Controls:")
    print("  'q' - Quit the application")
    print("  's' - Take a screenshot")
    print("  'h' - Toggle help text")
    print("  'e' - Toggle eye detection")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Variables for tracking statistics
    total_faces_detected = 0
    max_faces_in_frame = 0
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        
        # Flip the frame horizontally if configured
        if CONFIG['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size']
        )
        
        # Update statistics
        face_count = len(faces)
        total_faces_detected += face_count
        max_faces_in_frame = max(max_faces_in_frame, face_count)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                CONFIG['face_box_color'], 
                CONFIG['face_box_thickness']
            )
            
            # Detect eyes if configured
            if CONFIG['detect_eyes']:
                # Extract the region of interest (face area)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Draw rectangles around detected eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, 
                        (ex, ey), 
                        (ex + ew, ey + eh), 
                        CONFIG['eye_box_color'], 
                        CONFIG['eye_box_thickness']
                    )
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display information on the frame
        y_position = 30  # Starting y position for text
        
        # Display FPS
        if CONFIG['show_fps']:
            draw_text(frame, f"FPS: {fps:.1f}", (10, y_position))
            y_position += 30
        
        # Display face count
        draw_text(frame, f"Faces: {face_count}", (10, y_position))
        y_position += 30
        
        # Display max faces detected
        draw_text(frame, f"Max Faces: {max_faces_in_frame}", (10, y_position))
        y_position += 30
        
        # Display help text if configured
        if CONFIG['show_help']:
            help_y = CONFIG['frame_height'] - 120
            draw_text(frame, "Controls:", (10, help_y), size=0.5)
            draw_text(frame, "q: Quit  |  s: Screenshot", (10, help_y + 25), size=0.5)
            draw_text(frame, "h: Toggle Help  |  e: Toggle Eyes", (10, help_y + 50), size=0.5)
        
        # Display the resulting frame
        cv2.imshow('Advanced Face Tracker', frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Take screenshot
            filename = take_screenshot(frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('h'):  # Toggle help text
            CONFIG['show_help'] = not CONFIG['show_help']
        elif key == ord('e'):  # Toggle eye detection
            CONFIG['detect_eyes'] = not CONFIG['detect_eyes']
            status = "enabled" if CONFIG['detect_eyes'] else "disabled"
            print(f"Eye detection {status}")
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Display final statistics
    print("\nFace Tracker Statistics:")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Maximum faces in a single frame: {max_faces_in_frame}")
    print("Advanced Face Tracker stopped.")


if __name__ == "__main__":
    main()