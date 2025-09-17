import cv2
import numpy as np
import pyttsx3
from time import sleep
import warnings

warnings.filterwarnings("ignore")

class LightweightSmartGoggles:
    def __init__(self):
        print("Initializing Lightweight Smart Goggles...")
        
        # Initialize text-to-speech
        self.engine = None
        try:
            print("Initializing TTS engine...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', 1.0)
            print("Text-to-speech working correctly")
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.engine = None
        
        # Detection settings
        self.speak_delay = 60  # Speak every 2 seconds
        self.frame_count = 0
        
        # Edge detection parameters
        self.blur_kernel = 5
        self.canny_low = 50
        self.canny_high = 150
        self.min_contour_area = 1000
        self.close_distance_threshold = 8000  # Contour area for "close" objects
        
        print("Lightweight Smart Goggles initialized successfully!")
    
    def detect_obstacles(self, frame):
        """Detect obstacles using edge detection and contours"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            
            # Create display frame
            display_frame = frame.copy()
            
            # Process each contour
            obstacles = []
            for contour in significant_contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Estimate distance based on area
                distance_text, distance_meters = self.estimate_distance_from_area(area)
                
                # Get position
                position = self.get_position(x + w//2, frame.shape[1])
                
                # Color coding based on distance
                if distance_meters < 1.0:
                    color = (0, 0, 255)  # Red - very close
                elif distance_meters < 2.0:
                    color = (0, 165, 255)  # Orange - close
                elif distance_meters < 3.0:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 255, 0)  # Green - far
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw contour
                cv2.drawContours(display_frame, [contour], -1, color, 2)
                
                # Add label
                label = f"Obstacle {distance_text} {position}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store obstacle info
                obstacles.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'distance_meters': distance_meters,
                    'position': position,
                    'center_x': x + w//2
                })
            
            return obstacles, display_frame, edges
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], frame.copy(), np.zeros_like(frame[:,:,0])
    
    def estimate_distance_from_area(self, area):
        """Estimate distance based on contour area"""
        if area > 20000:
            return "0.5m", 0.5
        elif area > 15000:
            return "1.0m", 1.0
        elif area > 10000:
            return "1.5m", 1.5
        elif area > 8000:
            return "2.0m", 2.0
        elif area > 5000:
            return "3.0m", 3.0
        elif area > 3000:
            return "4.0m", 4.0
        elif area > 2000:
            return "5.0m", 5.0
        else:
            return "6.0m+", 6.0
    
    def get_position(self, center_x, frame_width):
        """Get position of object in frame"""
        if center_x < frame_width * 0.25:
            return "far left"
        elif center_x < frame_width * 0.45:
            return "left"
        elif center_x < frame_width * 0.55:
            return "center"
        elif center_x < frame_width * 0.75:
            return "right"
        else:
            return "far right"
    
    def is_obstacle_straight_ahead(self, obstacle, frame_width):
        """Check if obstacle is straight ahead"""
        center_x = obstacle['center_x']
        return frame_width * 0.4 <= center_x <= frame_width * 0.6
    
    def speak_simple(self, text):
        """Simple text-to-speech"""
        if not text or not self.engine:
            return False
            
        try:
            print(f"Speaking: {text}")
            self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            return False
    
    def create_obstacle_announcement(self, obstacles, frame):
        """Create announcements for close obstacles straight ahead"""
        if not obstacles:
            return None
        
        frame_width = frame.shape[1]
        warnings = []
        
        # Filter for close obstacles straight ahead
        for obstacle in obstacles:
            if (obstacle['distance_meters'] < 2.0 and 
                self.is_obstacle_straight_ahead(obstacle, frame_width)):
                distance_text = f"{obstacle['distance_meters']:.1f}m"
                warnings.append(f"Obstacle ahead at {distance_text}")
        
        # Return warning if any close obstacles
        if warnings:
            return ". ".join(warnings[:2])  # Limit to 2 warnings
        else:
            return None
    
    def add_status_overlay(self, frame, obstacles):
        """Add status information to frame"""
        height, width = frame.shape[:2]
        
        # Count close obstacles ahead
        close_ahead = sum(1 for obs in obstacles 
                         if obs['distance_meters'] < 2.0 and 
                         self.is_obstacle_straight_ahead(obs, width))
        
        # Status bar
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Obstacles: {len(obstacles)} | Close Ahead: {close_ahead}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count} | TTS: {'OK' if self.engine else 'FAILED'}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Warning if close obstacles
        if close_ahead > 0:
            cv2.putText(frame, f"*** WARNING: {close_ahead} OBSTACLES AHEAD ***", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def run(self):
        """Main loop"""
        # Initialize camera
        cap = None
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera found at index {camera_index}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("Camera not accessible!")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)            # Lower FPS to save CPU
        
        print("Lightweight Smart Goggles Active!")
        print("Features: Edge detection obstacle avoidance")
        print("Controls:")
        print("- 'q' or ESC: Quit")
        print("- 's': Manual obstacle check")
        print("- SPACE: Pause/Resume")
        print("- 'e': Show edge detection view")
        
        paused = False
        show_edges = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame capture failed")
                    break
                
                self.frame_count += 1
                
                if not paused:
                    # Detect obstacles
                    obstacles, display_frame, edges = self.detect_obstacles(frame)
                    
                    # Add status overlay
                    self.add_status_overlay(display_frame, obstacles)
                    
                    # Speech announcements (every 2 seconds)
                    if self.frame_count % self.speak_delay == 0:
                        announcement = self.create_obstacle_announcement(obstacles, frame)
                        if announcement:
                            print(f"Frame {self.frame_count}: OBSTACLE ALERT - {announcement}")
                            self.speak_simple(announcement)
                        else:
                            print(f"Frame {self.frame_count}: No close obstacles ahead")
                    
                    # Choose what to display
                    if show_edges:
                        # Convert edges to 3-channel for display
                        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                        cv2.putText(edge_display, "Edge Detection View", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Lightweight Smart Goggles', edge_display)
                    else:
                        cv2.imshow('Lightweight Smart Goggles', display_frame)
                
                else:
                    # Paused
                    paused_frame = frame.copy()
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Lightweight Smart Goggles', paused_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Quit
                    break
                    
                elif key == ord('s'):  # Manual check
                    if not paused:
                        obstacles, _, _ = self.detect_obstacles(frame)
                        announcement = self.create_obstacle_announcement(obstacles, frame)
                        if announcement:
                            print("Manual check:", announcement)
                            self.speak_simple(announcement)
                        else:
                            print("No obstacles ahead")
                    
                elif key == ord(' '):  # Pause/Resume
                    paused = not paused
                    print(f"Detection: {'PAUSED' if paused else 'RESUMED'}")
                
                elif key == ord('e'):  # Toggle edge view
                    show_edges = not show_edges
                    print(f"Edge view: {'ON' if show_edges else 'OFF'}")
        
        except KeyboardInterrupt:
            print("User interrupted")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        finally:
            print("Shutting down...")
            cap.release()
            cv2.destroyAllWindows()
            if self.engine:
                self.engine.stop()
            print("Lightweight Smart Goggles stopped")

def main():
    """Main function"""
    print("LIGHTWEIGHT SMART GOGGLES")
    print("=" * 40)
    print("Features:")
    print("- Edge detection obstacle detection")
    print("- Distance estimation")
    print("- Audio warnings for close obstacles")
    print("- Minimal resource usage")
    print("- No heavy ML libraries required")
    print("=" * 40)
    
    try:
        goggles = LightweightSmartGoggles()
        goggles.run()
    except Exception as e:
        print(f"Failed to start Smart Goggles: {e}")
        print("Required: pip install opencv-python pyttsx3 numpy")

if __name__ == "__main__":
    main()