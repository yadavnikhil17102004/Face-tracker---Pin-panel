# Face Tracker & ATM Security Prototype

A comprehensive face tracking application suite that includes basic face detection and an advanced ATM security prototype system. Uses webcam for real-time face detection and tracking with security monitoring capabilities.

This repository includes three applications:
1. **Simple Face Tracker** - Basic version with face detection only
2. **Advanced Face Tracker** - Extended version with eye detection, screenshots, and more features  
3. **ATM Security Prototype** - Demonstration security system for ATM environments

## Features

### Simple Face Tracker
- Real-time face detection using OpenCV
- FPS counter to monitor performance
- Face counter showing number of detected faces
- Configurable parameters for easy customization
- Mirror mode (horizontal flip) for intuitive interaction

### Advanced Face Tracker
All features from the simple version, plus:
- Eye detection within detected faces
- Screenshot capture functionality (press 's')
- Toggle help display (press 'h')
- Toggle eye detection on/off (press 'e')
- Enhanced statistics tracking
- Organized screenshot storage

### 🏧 ATM Security Prototype (NEW!)
An innovative security demonstration system featuring:
- **Dual Window Interface**: Separate camera monitoring and keypad simulation windows
- **Multi-Person Detection**: Automatic warning when more than one person is detected
- **Security Alerts**: Visual and audio warnings during security breaches
- **Automatic Screenshots**: Captures security incidents for review
- **Transaction Blocking**: Prevents PIN entry during security breaches
- **Real-time Status**: Live security status updates in both windows
- **Professional UI**: ATM-style keypad with realistic design

**Security Features:**
- 🔴 Red warning boxes when multiple people detected
- 🟢 Green secure status for single user
- 📸 Automatic screenshot capture during breaches
- 🔊 Audio alert system
- ⏱️ Timed warning periods
- 📊 Security breach statistics

## Requirements

- Python 3.6 or higher
- OpenCV library
- Pygame (for ATM security audio alerts)
- Tkinter (for ATM keypad interface - usually included with Python)
- A working webcam

## Installation

1. Clone or download this repository

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Simple Face Tracker
Run the simple face tracker application:

```bash
python face_tracker.py
```

Or use the provided batch file (Windows):

```
run_face_tracker.bat
```

Press 'q' to quit the application.

### Advanced Face Tracker
Run the advanced face tracker application:

```bash
python advanced_face_tracker.py
```

Or use the provided batch file (Windows):

```
run_advanced_face_tracker.bat
```

**Controls:**
- Press 'q' to quit the application
- Press 's' to take a screenshot
- Press 'h' to toggle help text display
- Press 'e' to toggle eye detection on/off

### 🏧 ATM Security Prototype
Run the ATM security prototype:

```bash
python atm_security_prototype.py
```

Or use the provided batch file (Windows):

```
run_atm_security.bat
```

**Controls:**
- Press 'q' to quit the camera application
- Press 's' to take a manual screenshot
- Press 'k' to open/close the keypad window
- Press 'h' to toggle help display

**How it Works:**
1. **Camera Window**: Shows live feed with face detection
   - Green boxes = Secure (1 person or less)
   - Red boxes = Security breach (multiple people)
   - Real-time security status display
   
2. **Keypad Window**: Simulates ATM PIN entry
   - Numeric keypad for PIN entry
   - Security status indicator
   - Transaction blocking during breaches
   - Warning popups for security alerts

**Security Demo Scenarios:**
- 👤 **Single Person**: Normal operation, green indicators, PIN entry allowed
- 👥 **Multiple People**: Security breach mode, red warnings, PIN entry blocked
- 📸 **Auto Documentation**: Screenshots automatically saved during security events
- 🔊 **Audio Alerts**: Warning sounds when multiple people detected

## Configuration

### Simple Face Tracker
You can customize the simple face tracker by modifying the `CONFIG` dictionary in the `face_tracker.py` file:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `camera_id` | Camera device ID | 0 (built-in webcam) |
| `frame_width` | Width of camera frame | 640 |
| `frame_height` | Height of camera frame | 480 |
| `box_color` | Color of face bounding box (BGR format) | (0, 255, 0) (green) |
| `box_thickness` | Thickness of bounding box lines | 2 |
| `min_face_size` | Minimum face size to detect | (30, 30) |
| `scale_factor` | How much the image size is reduced at each scale | 1.1 |
| `min_neighbors` | How many neighbors each candidate rectangle should have | 5 |
| `show_fps` | Whether to display FPS counter | True |
| `flip_horizontal` | Flip the camera horizontally (mirror mode) | True |

### Advanced Face Tracker
The advanced face tracker has additional configuration options in the `CONFIG` dictionary in the `advanced_face_tracker.py` file:

#### Camera Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `camera_id` | Camera device ID | 0 (built-in webcam) |
| `frame_width` | Width of camera frame | 640 |
| `frame_height` | Height of camera frame | 480 |
| `flip_horizontal` | Flip the camera horizontally (mirror mode) | True |

#### Face Detection Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `face_box_color` | Color of face bounding box (BGR format) | (0, 255, 0) (green) |
| `face_box_thickness` | Thickness of bounding box lines | 2 |
| `min_face_size` | Minimum face size to detect | (30, 30) |
| `scale_factor` | How much the image size is reduced at each scale | 1.1 |
| `min_neighbors` | How many neighbors each candidate rectangle should have | 5 |

#### Eye Detection Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `detect_eyes` | Whether to detect eyes within faces | True |
| `eye_box_color` | Color of eye bounding box (BGR format) | (255, 0, 0) (blue) |
| `eye_box_thickness` | Thickness of eye bounding box lines | 1 |

#### Display Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `show_fps` | Whether to display FPS counter | True |
| `show_help` | Whether to display help text | True |
| `text_color` | Color of text overlays (BGR format) | (0, 0, 255) (red) |
| `text_size` | Size of text overlays | 0.7 |
| `text_thickness` | Thickness of text overlays | 2 |

#### Screenshot Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `screenshot_dir` | Directory to save screenshots | 'screenshots' |
| `screenshot_format` | Format to save screenshots (jpg or png) | 'jpg' |

### ATM Security Prototype Configuration
The ATM security system can be customized by modifying the `CONFIG` dictionary in `atm_security_prototype.py`:

#### Security Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `max_safe_people` | Maximum safe number of people | 1 |
| `warning_duration` | Duration to show warning (seconds) | 3 |
| `auto_screenshot` | Auto screenshot on security breach | True |
| `face_box_color` | Color for secure status (BGR) | (0, 255, 0) (green) |
| `warning_box_color` | Color for warning status (BGR) | (0, 0, 255) (red) |

#### Output Directories
- **Screenshots**: `screenshots/` (manual screenshots)
- **Security Screenshots**: `security_screenshots/` (automatic security breach captures)

## How It Works

### Face Detection
All versions use the Haar Cascade classifier from OpenCV, which is a machine learning-based approach where a cascade function is trained from many positive and negative images. It's particularly efficient for face detection.

### ATM Security System
The security prototype implements a multi-threaded approach:
1. **Camera Thread**: Continuously monitors for faces and security breaches
2. **GUI Thread**: Manages the keypad interface and user interactions
3. **Security Logic**: Coordinates between camera detection and transaction blocking

The basic process for both versions:
1. Captures video from your webcam
2. Converts each frame to grayscale
3. Applies the face detection algorithm
4. Draws rectangles around detected faces
5. Displays the processed frame with face counts and FPS

### Eye Detection (Advanced Version)
The advanced version also uses a Haar Cascade classifier specifically trained for eye detection. For each detected face:
1. A region of interest (ROI) is extracted from the face area
2. The eye detection algorithm is applied to this region
3. Rectangles are drawn around detected eyes

### Screenshots (Advanced Version)
The advanced version allows you to capture screenshots by pressing the 's' key:
1. Screenshots are saved to the 'screenshots' directory (created automatically)
2. Files are named with a timestamp (e.g., 'face_tracker_20230615_120530.jpg')
3. The format can be configured (JPG or PNG)

## Troubleshooting

- If the webcam doesn't open, check that your camera is connected and not being used by another application
- If face detection is not working well, try adjusting the `scale_factor` and `min_neighbors` parameters
- For better performance on slower computers, try reducing the `frame_width` and `frame_height`

## Further Extensions

Some ideas for further extending this application:

- Add face recognition to identify specific people
- Track faces across frames to assign consistent IDs
- Add emotion detection to recognize facial expressions
- Implement facial landmark detection (nose, mouth, etc.)
- Add age and gender estimation
- Create a recording feature to save video
- Implement motion detection to only track when movement occurs
- Add a GUI for adjusting settings without modifying code

## License

This project is open source and available for personal and educational use.