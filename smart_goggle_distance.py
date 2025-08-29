import cv2
import pyttsx3
import torch
import numpy as np
from time import sleep
import warnings

warnings.filterwarnings("ignore")

class FixedSmartGoggles:
    def __init__(self):
        print("Initializing Smart Goggles...")
        
        try:
            print("Loading YOLOv5 model...")
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', 
                                           force_reload=True, trust_repo=True)
            print("YOLOv5 loaded successfully")
            
        except Exception as e:
            print(f"YOLOv5 loading failed: {e}")
            raise Exception("Could not load YOLO model")
        
        # Initialize text-to-speech with better error handling
        self.engine = None
        try:
            print("Initializing TTS engine...")
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', 140)  # Slower rate
            self.engine.setProperty('volume', 1.0)
            
            # Test the engine
            print("Testing TTS engine...")
            self.engine.say("TTS engine initialized")
            self.engine.runAndWait()
            
            print("Text-to-speech working correctly")
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            print("Trying alternative TTS initialization...")
            try:
                import platform
                if platform.system() == "Windows":
                    self.engine = pyttsx3.init('sapi5')
                else:
                    self.engine = pyttsx3.init('espeak')
                
                self.engine.setProperty('rate', 140)
                self.engine.setProperty('volume', 1.0)
                print("Alternative TTS initialized")
            except Exception as e2:
                print(f"Alternative TTS also failed: {e2}")
                self.engine = None
        
        # Detection settings
        self.speak_delay = 90  # Speak every 3 seconds to give TTS time
        self.frame_count = 0
        self.confidence_threshold = 0.4
        
        # Removed priority objects - detect all objects equally
        
        # Simple tracking for voice
        self.last_speech_frame = 0
        
        print("Smart Goggles initialized successfully!")
    
    def detect_objects(self, frame):
        """Object detection with better processing"""
        try:
            results = self.yolo_model(frame)
            detections = results.pandas().xyxy[0]
            detections = detections[detections['confidence'] >= self.confidence_threshold]
            
            display_frame = frame.copy()
            
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                conf = detection['confidence']
                label = detection['name']
                
                # Calculate distance in meters
                distance_text, distance_meters = self.estimate_distance([x1, y1, x2, y2], label)
                
                # Color code based on distance
                if distance_meters < 1.0:
                    color = (0, 0, 255)  # Red - very close
                elif distance_meters < 2.0:
                    color = (0, 165, 255)  # Orange - close
                elif distance_meters < 4.0:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 255, 0)  # Green - far
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Get position
                position = self.get_position([x1, y1, x2, y2], frame.shape[1])
                
                # Prepare label text
                label_text = f"{label} {distance_text} {position}"
                
                # Calculate label size and position
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(display_frame, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), 
                             color, -1)
                
                # Draw label text
                cv2.putText(display_frame, label_text, 
                           (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return detections, display_frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            import pandas as pd
            return pd.DataFrame(), frame.copy()
    
    def estimate_distance(self, bbox, object_class):
        """More accurate distance estimation based on real-world object sizes"""
        x1, y1, x2, y2 = bbox
        box_height = y2 - y1
        box_width = x2 - x1
        box_area = box_height * box_width
        
        # More realistic distance estimation
        if object_class == 'person':
            # Average person height ~1.7m
            if box_height > 400:
                return "0.8m", 0.8
            elif box_height > 280:
                return "1.5m", 1.5
            elif box_height > 200:
                return "2.5m", 2.5
            elif box_height > 140:
                return "4.0m", 4.0
            else:
                return "6.0m", 6.0
                
        elif object_class in ['car', 'truck', 'bus']:
            if box_area > 120000:
                return "2.0m", 2.0
            elif box_area > 80000:
                return "4.0m", 4.0
            elif box_area > 40000:
                return "8.0m", 8.0
            elif box_area > 20000:
                return "12.0m", 12.0
            else:
                return "20.0m", 20.0
                
        elif object_class == 'motorcycle':
            if box_area > 60000:
                return "1.5m", 1.5
            elif box_area > 30000:
                return "3.0m", 3.0
            elif box_area > 15000:
                return "6.0m", 6.0
            else:
                return "10.0m", 10.0
                
        elif object_class in ['cell phone', 'remote', 'mouse', 'keyboard']:
            # Small handheld objects
            if box_area > 8000:
                return "0.3m", 0.3
            elif box_area > 4000:
                return "0.6m", 0.6
            elif box_area > 2000:
                return "1.0m", 1.0
            else:
                return "1.5m", 1.5
                
        elif object_class in ['chair', 'couch']:
            # Typical chair height ~0.9m
            if box_height > 300:
                return "1.0m", 1.0
            elif box_height > 200:
                return "2.0m", 2.0
            elif box_height > 120:
                return "3.5m", 3.5
            elif box_height > 80:
                return "5.0m", 5.0
            else:
                return "8.0m", 8.0
                
        elif object_class in ['bed', 'dining table']:
            # Table/bed height ~0.7-0.8m
            if box_area > 60000:
                return "1.2m", 1.2
            elif box_area > 35000:
                return "2.5m", 2.5
            elif box_area > 18000:
                return "4.0m", 4.0
            elif box_area > 8000:
                return "6.0m", 6.0
            else:
                return "10.0m", 10.0
                
        elif object_class in ['bottle', 'cup']:
            if box_height > 80:
                return "0.5m", 0.5
            elif box_height > 50:
                return "1.0m", 1.0
            elif box_height > 30:
                return "2.0m", 2.0
            else:
                return "3.0m", 3.0
        else:
            # Generic estimation based on area
            if box_area > 80000:
                return "1.5m", 1.5
            elif box_area > 40000:
                return "3.0m", 3.0
            elif box_area > 20000:
                return "5.0m", 5.0
            elif box_area > 10000:
                return "8.0m", 8.0
            else:
                return "12.0m", 12.0
    
    def get_position(self, bbox, frame_width):
        """Enhanced position detection"""
        x1, x2 = bbox[0], bbox[2]
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width * 0.2:
            return "far left"
        elif center_x < frame_width * 0.4:
            return "left"
        elif center_x < frame_width * 0.6:
            return "center"
        elif center_x < frame_width * 0.8:
            return "right"
        else:
            return "far right"
    
    def speak_simple(self, text):
        """Improved TTS with better error handling and debugging"""
        if not text:
            return False
            
        print(f"Attempting to speak: {text}")
        
        if not self.engine:
            print("TTS engine not available - text only")
            return False
            
        try:
            # Clear any pending speech
            self.engine.stop()
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            print("Speech completed successfully")
            return True
            
        except Exception as e:
            print(f"TTS error during speech: {e}")
            
            # Try to reinitialize engine
            try:
                print("Attempting to reinitialize TTS...")
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 140)
                self.engine.setProperty('volume', 1.0)
                
                self.engine.say(text)
                self.engine.runAndWait()
                print("Speech successful after reinit")
                return True
                
            except Exception as e2:
                print(f"Reinit also failed: {e2}")
                return False
    
    def create_simple_announcement(self, detections, frame):
        """Create simple announcements for all objects"""
        if len(detections) == 0:
            return "No objects detected"
        
        frame_height, frame_width = frame.shape[:2]
        announcements = []
        
        # Get all detections
        for _, detection in detections.iterrows():
            obj_name = detection['name']
            bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
            distance_text, distance_meters = self.estimate_distance(bbox, obj_name)
            position = self.get_position(bbox, frame_width)
            
            # Simple announcement for each object
            announcements.append(f"{obj_name} {distance_text} {position}")
        
        # Limit to 3 closest objects to avoid too much speech
        return ". ".join(announcements[:3])
    
    def speak_threaded(self, text):
        """Simple wrapper for speak - removed threading complexity"""
        return self.speak_simple(text)
    
    def run(self):
        """Main loop with proper exit handling"""
        # Try different camera indices
        cap = None
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera found at index {camera_index}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("Camera not accessible!")
            print("Troubleshooting:")
            print("1. Check if camera is connected")
            print("2. Close other apps using camera")
            print("3. Check camera permissions")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Smart Goggles Active!")
        print("Controls:")
        print("- 'q' or ESC: Quit")
        print("- 's': Immediate speech (instant scan)")
        print("- SPACE: Pause/Resume")
        
        paused = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame capture failed")
                    break
                
                self.frame_count += 1
                
                if not paused:
                    # Object detection
                    detections, display_frame = self.detect_objects(frame)
                    
                    # Add status overlay
                    self.add_status_overlay(display_frame, detections)
                    
                    # Speech announcements every 3 seconds
                    if self.frame_count % self.speak_delay == 0 and len(detections) > 0:
                        announcement = self.create_simple_announcement(detections, frame)
                        print(f"Frame {self.frame_count}: Triggering speech")
                        success = self.speak_simple(announcement)
                        if not success:
                            print("Speech failed - continuing with visual only")
                
                else:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Smart Goggles', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    print("Quit command received")
                    break
                    
                elif key == ord('s'):  # Manual immediate speech
                    print("Manual speech triggered")
                    if not paused:
                        detections, _ = self.detect_objects(frame)
                        announcement = self.create_simple_announcement(detections, frame)
                        print("Manual announcement:", announcement)
                        self.speak_simple(announcement)
                    
                elif key == ord(' '):  # Pause/Resume
                    paused = not paused
                    print(f"Detection: {'PAUSED' if paused else 'RESUMED'}")
        
        except KeyboardInterrupt:
            print("User interrupted")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("Shutting down...")
            cap.release()
            cv2.destroyAllWindows()
            if self.engine:
                self.engine.stop()
            print("Smart Goggles stopped")
    
    def add_status_overlay(self, frame, detections):
        """Add minimal status information"""
        height, width = frame.shape[:2]
        
        # Simple status bar
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, 80), (100, 100, 100), 1)
        
        cv2.putText(frame, f"Objects: {len(detections)} | TTS: {'OK' if self.engine else 'FAILED'}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {self.frame_count} | Next speech: {self.speak_delay - (self.frame_count % self.speak_delay)}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame, "Controls: q=quit, s=immediate speak, space=pause", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    """Main function"""
    print("SMART GOGGLES - AI Navigation Assistant")
    print("=" * 50)
    print("Features:")
    print("- Real-time object detection")
    print("- Distance estimation in meters") 
    print("- Smart speech alerts")
    print("- Reduced repetitive announcements")
    print("=" * 50)
    
    try:
        goggles = FixedSmartGoggles()
        goggles.run()
    except Exception as e:
        print(f"Failed to start Smart Goggles: {e}")
        print("Solutions:")
        print("1. Restart terminal")
        print("2. Check camera permissions")
        print("3. Install dependencies: pip install opencv-python torch ultralytics pyttsx3")

if __name__ == "__main__":
    main()


    