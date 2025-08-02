"""
Demo Instructions for ATM Security Prototype

This file contains step-by-step instructions for demonstrating
the ATM Security Prototype functionality.
"""

def print_demo_instructions():
    print("=" * 60)
    print("🏧 ATM SECURITY PROTOTYPE - DEMO INSTRUCTIONS")
    print("=" * 60)
    print()
    
    print("📋 DEMO SCENARIO GUIDE:")
    print()
    
    print("1️⃣  SINGLE USER SCENARIO (Normal Operation)")
    print("   • Position yourself alone in front of the camera")
    print("   • Notice: Green bounding box around your face")
    print("   • Keypad shows: '🟢 SECURE - Single User Detected'")
    print("   • Try entering PIN on keypad - it should work normally")
    print("   • Take manual screenshot with 's' key")
    print()
    
    print("2️⃣  MULTIPLE USERS SCENARIO (Security Breach)")
    print("   • Have a second person enter camera view")
    print("   • Notice: Red bounding boxes appear")
    print("   • Camera shows: '⚠️ SECURITY BREACH - X PEOPLE DETECTED!'")
    print("   • Keypad shows: '🔴 SECURITY BREACH - Multiple Users Detected!'")
    print("   • Audio warning sound plays")
    print("   • Try entering PIN - it will be blocked with warning")
    print("   • Auto screenshot is taken and saved")
    print()
    
    print("3️⃣  RECOVERY SCENARIO")
    print("   • Have extra person leave camera view")
    print("   • Wait for warning timer to expire (3 seconds)")
    print("   • Notice: System returns to secure green status")
    print("   • PIN entry is re-enabled on keypad")
    print()
    
    print("🎮 CONTROLS SUMMARY:")
    print("   Camera Window:")
    print("   • 'q' - Quit application")
    print("   • 's' - Take manual screenshot")
    print("   • 'k' - Open/Close keypad window")
    print("   • 'h' - Toggle help display")
    print()
    print("   Keypad Window:")
    print("   • Number buttons - Enter digits")
    print("   • CLEAR - Clear entered PIN")
    print("   • ENTER - Submit PIN (when 4 digits entered)")
    print()
    
    print("📁 OUTPUT FILES:")
    print("   • Manual screenshots: screenshots/")
    print("   • Security screenshots: security_screenshots/")
    print("   • Naming: security_breach_YYYYMMDD_HHMMSS.jpg")
    print()
    
    print("💡 DEMO TIPS:")
    print("   • Position windows side-by-side for best effect")
    print("   • Use good lighting for better face detection")
    print("   • Move slowly to avoid detection glitches")
    print("   • Try different angles and distances")
    print("   • Test with various numbers of people")
    print()
    
    print("🔧 TROUBLESHOOTING:")
    print("   • If camera doesn't start: Check camera permissions")
    print("   • If no face detected: Improve lighting, check position")
    print("   • If keypad doesn't open: Check tkinter installation")
    print("   • If no sound: Check pygame installation and audio settings")
    print()
    
    print("=" * 60)
    print("Ready to start demo? Run: run_atm_security.bat")
    print("=" * 60)

if __name__ == "__main__":
    print_demo_instructions()
