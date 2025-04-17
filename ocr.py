import pytesseract
from PIL import ImageGrab
import pyperclip
import sys
import time
import signal
import threading
from PIL import Image

def image_to_text(monitor):
    last_error_time = 0
    error_cooldown = 60  # Wait 60 seconds before showing same error again
    
    try:
        while monitor['running']:
            try:
                # Get content from clipboard
                clipboard_content = ImageGrab.grabclipboard()
                
                # Only process if it's an image
                if isinstance(clipboard_content, Image.Image):
                    # Convert image to text
                    text = pytesseract.image_to_string(clipboard_content)
                    
                    if text.strip():
                        # Copy to clipboard
                        pyperclip.copy(text)
                        print("\nText extracted and copied to clipboard!")
                        print(f"\nText: {text}")
                
                # If it's not an image, just continue monitoring
                time.sleep(0.5)
                
            except Exception as e:
                current_time = time.time()
                # Only show error message if we haven't shown one recently
                if current_time - last_error_time > error_cooldown:
                    print(f"\nNotice: Skipping non-image clipboard content")
                    last_error_time = current_time
                
                # Brief pause before continuing
                time.sleep(0.5)
                continue
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print("\nStopping OCR monitor...")
    monitor['running'] = False
    sys.exit(0)

def main():
    # Check if Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Error: Tesseract is not installed or not in PATH.")
        print("Please install Tesseract OCR first:")
        print("- Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    
    print("OCR Monitor Running")
    print("Use Win+Shift+S to capture screenshots")
    print("Press Ctrl+C to stop")
    print("\nMonitoring clipboard for images...")
    
    # Set up Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Use a dictionary to share the running state between threads
    global monitor
    monitor = {'running': True}
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=image_to_text, args=(monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Keep the main thread alive but responsive to Ctrl+C
    try:
        while monitor['running']:
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor['running'] = False
        print("\nStopping OCR monitor...")
        sys.exit(0)

if __name__ == "__main__":
    main()