import cv2
import time

"""
Simple Face Tracking Application

This program uses OpenCV to detect faces in a webcam stream and draw bounding boxes around them.
It's designed to be efficient and easy to understand with minimal dependencies.

Configuration parameters are at the top of the file for easy customization.
"""

# Configuration parameters - modify these to customize the application
CONFIG = {
    'camera_id': 0,                # Camera ID (usually 0 for built-in webcam)
    'frame_width': 640,            # Width of the camera frame
    'frame_height': 480,           # Height of the camera frame
    'box_color': (0, 255, 0),      # Color of face bounding box (BGR format)
    'box_thickness': 2,            # Thickness of bounding box lines
    'min_face_size': (30, 30),     # Minimum face size to detect
    'scale_factor': 1.1,           # How much the image size is reduced at each image scale
    'min_neighbors': 5,            # How many neighbors each candidate rectangle should have
    'show_fps': True,              # Whether to display FPS counter
    'flip_horizontal': True,       # Flip the camera horizontally (mirror mode)
}


def main():
    # Load the pre-trained face detector from OpenCV
    # This uses the Haar Cascade classifier which is efficient for real-time applications
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Check if the face cascade loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    # Initialize the webcam
    cap = cv2.VideoCapture(CONFIG['camera_id'])
    
    # Set the frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face tracker started. Press 'q' to quit.")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
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
        # This improves performance and is required for Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size']
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                CONFIG['box_color'], 
                CONFIG['box_thickness']
            )
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on the frame if configured
        if CONFIG['show_fps']:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame, 
                fps_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
        
        # Display the number of faces detected
        faces_text = f"Faces: {len(faces)}"
        cv2.putText(
            frame, 
            faces_text, 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # Display the resulting frame
        cv2.imshow('Face Tracker', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Face tracker stopped.")


if __name__ == "__main__":
    main()