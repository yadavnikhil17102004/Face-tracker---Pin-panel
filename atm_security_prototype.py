import cv2
import time
import os
import tkinter as tk
from tkinter import messagebox
import threading
from datetime import datetime
import pygame
from queue import Queue
import numpy as np

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
    'frame_width': 640,
    'frame_height': 480,
    'flip_horizontal': True,
    
    # Face detection settings
    'face_box_color': (0, 255, 0),  # Green for single person
    'warning_box_color': (0, 0, 255),  # Red for multiple people
    'face_box_thickness': 3,
    'min_face_size': (30, 30),
    'scale_factor': 1.1,
    'min_neighbors': 5,
    
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
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Security state variables
        self.security_breach = False
        self.warning_start_time = 0
        self.breach_count = 0
        
        # Initialize camera
        self.cap = None
        
        # Create warning sound
        self.create_warning_sound()
        
        # Ensure screenshot directory exists
        self.ensure_dir(CONFIG['screenshot_dir'])
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.show_help = True
    
    def create_warning_sound(self):
        """Create a simple warning beep sound"""
        try:
            # Create a simple beep sound using pygame
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            # Create a simple beep sound
            buffer = np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate).astype(np.float32)
            buffer = (buffer * 32767).astype(np.int16)
            
            # Convert to stereo
            stereo_buffer = np.column_stack((buffer, buffer))
            
            # Create sound object
            sound = pygame.mixer.Sound(stereo_buffer)
            self.warning_sound = sound
        except Exception as e:
            print(f"Warning: Could not create warning sound: {e}")
            self.warning_sound = None
    
    def ensure_dir(self, directory):
        """Ensure that a directory exists, creating it if necessary."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def take_screenshot(self, frame, reason="security_breach"):
        """Save a screenshot for security purposes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CONFIG['screenshot_dir']}/{reason}_{timestamp}.{CONFIG['screenshot_format']}"
        cv2.imwrite(filename, frame)
        return filename
    
    def play_warning_sound(self):
        """Play warning sound if available."""
        if self.warning_sound:
            try:
                self.warning_sound.play()
            except Exception as e:
                print(f"Error playing sound: {e}")
    
    def check_security_breach(self, face_count):
        """Check if there's a security breach and handle accordingly."""
        if face_count > CONFIG['max_safe_people']:
            if not self.security_breach:
                # New security breach detected
                self.security_breach = True
                self.warning_start_time = time.time()
                self.breach_count += 1
                self.play_warning_sound()
                print(f"⚠️  SECURITY ALERT: {face_count} people detected! ⚠️")
                
                # Notify the UI thread about the security breach
                self.event_queue.put(("security_breach", face_count))
        else:
            # Check if warning period has elapsed
            if self.security_breach and (time.time() - self.warning_start_time) > CONFIG['warning_duration']:
                self.security_breach = False
                print("✅ Security status: Normal")
                
                # Notify the UI thread that the breach is over
                self.event_queue.put(("security_normal", face_count))
    
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
        print("  - Real-time face detection")
        print("  - Multiple person warning system")
        print("  - Automatic security screenshots")
        print("  - Virtual keypad simulation")
        print("\nControls:")
        print("  'q' - Quit application")
        print("  's' - Manual screenshot")
        print("  'k' - Open/Close keypad window")
        print("  'h' - Toggle help display")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CONFIG['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.event_queue.put(("error", "Could not open camera"))
            return
        
        # Reset FPS counter
        self.frame_count = 0
        self.start_time = time.time()
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            if CONFIG['flip_horizontal']:
                frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=CONFIG['scale_factor'],
                minNeighbors=CONFIG['min_neighbors'],
                minSize=CONFIG['min_face_size']
            )
            
            face_count = len(faces)
            
            # Update UI about face count
            self.event_queue.put(("face_count", face_count))
            
            # Check for security breach
            self.check_security_breach(face_count)
            
            # Choose box color based on security status
            box_color = CONFIG['warning_box_color'] if self.security_breach else CONFIG['face_box_color']
            
            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, CONFIG['face_box_thickness'])
                
                # Add face number
                person_index = list(map(tuple, faces)).index((x, y, w, h)) + 1
                cv2.putText(frame, f"Person {person_index}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
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
            
            # Additional info
            if CONFIG['show_fps']:
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, CONFIG['text_color'], 2)
                y_pos += 25
            
            cv2.putText(frame, f"Security Breaches: {self.breach_count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG['text_color'], 2)
            
            # Help display
            if self.show_help:
                help_y = CONFIG['frame_height'] - 100
                cv2.putText(frame, "Controls: q=Quit | s=Screenshot | k=Keypad | h=Help", 
                           (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
            
            # Display frame
            cv2.imshow('ATM Security Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.event_queue.put(("quit", None))
                break
            elif key == ord('s'):
                filename = self.take_screenshot(frame, "manual")
                print(f"📸 Manual screenshot saved: {filename}")
                self.event_queue.put(("manual_screenshot", filename))
            elif key == ord('k'):
                self.event_queue.put(("toggle_keypad", None))
            elif key == ord('h'):
                self.show_help = not self.show_help
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        print(f"\n🏧 ATM Security System Stopped")
        print(f"Total security breaches detected: {self.breach_count}")


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
        
        # Schedule next event processing
        self.root.after(100, self.process_events)
    
    def quit(self):
        """Quit the application."""
        print("Shutting down ATM Security System...")
        
        # Stop the security monitor
        self.security_monitor.stop()
        
        # Destroy all windows
        self.root.quit()


def main():
    # Create root window
    root = tk.Tk()
    
    # Create and start application
    app = ATMSecurityApp(root)
    
    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
    
    def show_security_warning(self):
        """Show security warning in keypad window."""
        if self.keypad_window:
            # Schedule the warning to be shown in the main thread
            self.keypad_window.after(0, self._show_warning_dialog)
    
    def _show_warning_dialog(self):
        """Show the warning dialog in the main thread."""
        try:
            messagebox.showwarning(
                "SECURITY ALERT", 
                "⚠️ MULTIPLE PEOPLE DETECTED! ⚠️\n\nPlease ensure you are alone\nbefore entering your PIN.\n\nTransaction will be paused for security."
            )
        except:
            print("Warning: Could not show security dialog")
    
    def create_keypad_window(self):
        """Create the ATM keypad simulation window."""
        self.keypad_window = tk.Tk()
        self.keypad_window.title("ATM Security Prototype - PIN Entry")
        self.keypad_window.geometry("400x600")
        self.keypad_window.configure(bg='#2c3e50')
        
        # Title
        title_label = tk.Label(
            self.keypad_window, 
            text="🏧 ATM SECURITY PROTOTYPE", 
            font=('Arial', 16, 'bold'), 
            bg='#2c3e50', 
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Security status display
        self.security_status = tk.Label(
            self.keypad_window,
            text="🟢 SECURE - Single User Detected",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='#27ae60'
        )
        self.security_status.pack(pady=10)
        
        # PIN display
        self.pin_display = tk.Label(
            self.keypad_window,
            text="Enter PIN: ****",
            font=('Arial', 14),
            bg='#34495e',
            fg='white',
            relief='sunken',
            width=20,
            height=2
        )
        self.pin_display.pack(pady=20)
        
        # Keypad frame
        keypad_frame = tk.Frame(self.keypad_window, bg='#2c3e50')
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
        action_frame = tk.Frame(self.keypad_window, bg='#2c3e50')
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
            self.keypad_window,
            text="This is a PROTOTYPE demonstration.\nSecurity camera monitors for multiple users.\nTransaction is blocked if breach detected.",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7',
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
        
        # Start updating security status
        self.update_security_status()
        
        self.keypad_window.protocol("WM_DELETE_WINDOW", self.on_keypad_close)
    
    def update_security_status(self):
        """Update the security status display in keypad."""
        if self.keypad_window:
            if self.security_breach:
                self.security_status.config(
                    text="🔴 SECURITY BREACH - Multiple Users Detected!",
                    fg='#e74c3c'
                )
            else:
                self.security_status.config(
                    text="🟢 SECURE - Single User Detected",
                    fg='#27ae60'
                )
            
            # Schedule next update
            self.keypad_window.after(500, self.update_security_status)
    
    def keypad_press(self, key):
        """Handle keypad button press."""
        if self.security_breach:
            messagebox.showwarning("Security Alert", "Transaction blocked due to security breach!")
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
            messagebox.showwarning("Security Alert", "Transaction blocked due to security breach!")
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
    
    def on_keypad_close(self):
        """Handle keypad window close."""
        self.keypad_window.destroy()
        self.keypad_window = None
    
    def start_keypad(self):
        """Start the keypad window."""
        if not self.keypad_window:
            self.create_keypad_window()
            print("🔢 Keypad window opened")
    
    def run_camera_detection(self):
        """Main camera detection loop."""
        print("🏧 ATM Security System Started")
        print("Features:")
        print("  - Real-time face detection")
        print("  - Multiple person warning system")
        print("  - Automatic security screenshots")
        print("  - Virtual keypad simulation")
        print("\nControls:")
        print("  'q' - Quit application")
        print("  's' - Manual screenshot")
        print("  'k' - Open/Close keypad window")
        print("  'h' - Toggle help display")
        
        show_help = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            if CONFIG['flip_horizontal']:
                frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=CONFIG['scale_factor'],
                minNeighbors=CONFIG['min_neighbors'],
                minSize=CONFIG['min_face_size']
            )
            
            face_count = len(faces)
            
            # Check for security breach
            self.check_security_breach(face_count)
            
            # Choose box color based on security status
            box_color = CONFIG['warning_box_color'] if self.security_breach else CONFIG['face_box_color']
            
            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, CONFIG['face_box_thickness'])
                
                # Add face number
                cv2.putText(frame, f"Person {faces.tolist().index([x, y, w, h]) + 1}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Auto screenshot on security breach
            if self.security_breach and CONFIG['auto_screenshot']:
                if time.time() - self.warning_start_time < 0.5:  # Screenshot once per breach
                    filename = self.take_screenshot(frame)
                    print(f"🔴 Security screenshot saved: {filename}")
            
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
            
            # Additional info
            if CONFIG['show_fps']:
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, CONFIG['text_color'], 2)
                y_pos += 25
            
            cv2.putText(frame, f"Security Breaches: {self.breach_count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG['text_color'], 2)
            
            # Help display
            if show_help:
                help_y = CONFIG['frame_height'] - 100
                cv2.putText(frame, "Controls: q=Quit | s=Screenshot | k=Keypad | h=Help", 
                           (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['text_color'], 1)
            
            # Display frame
            cv2.imshow('ATM Security Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = self.take_screenshot(frame, "manual")
                print(f"📸 Manual screenshot saved: {filename}")
            elif key == ord('k'):
                if not self.keypad_window:
                    self.start_keypad()
                    print("🔢 Keypad window opened")
                else:
                    print("ℹ️  Keypad window already open")
            elif key == ord('h'):
                show_help = not show_help
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.keypad_window:
            self.keypad_window.destroy()
        
        print(f"\n🏧 ATM Security System Stopped")
        print(f"Total security breaches detected: {self.breach_count}")

def main():
    # Create root window
    root = tk.Tk()
    
    # Create and start application
    app = ATMSecurityApp(root)
    
    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
