import subprocess
from pathlib import Path
import pyttsx3
import torch
import warnings
import time

warnings.filterwarnings("ignore")


def capture_frame(filename="frame.jpg", width=640, height=480):
    """Capture a single frame from Pi camera using libcamera-still (headless)"""
    frame_path = Path(filename)
    subprocess.run([
        "libcamera-still",
        "-o", str(frame_path),
        "-n",  # no preview
        "--width", str(width),
        "--height", str(height)
    ], check=True)
    return str(frame_path)


class FixedSmartGogglesHeadless:
    def __init__(self):
        print("Initializing Smart Goggles (Headless)...")
        
        # Load YOLOv5
        try:
            print("Loading YOLOv5 model...")
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n',
                                             force_reload=True, trust_repo=True)
            print("YOLOv5 loaded successfully")
        except Exception as e:
            print(f"YOLOv5 loading failed: {e}")
            raise Exception("Could not load YOLO model")

        # Initialize TTS engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', 1.0)
            print("TTS initialized")
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.engine = None

        # Detection settings
        self.confidence_threshold = 0.4
        self.frame_count = 0
        self.speak_delay = 90  # announce every 90 frames (~3 sec at 30fps)
        
        print("Smart Goggles Headless Initialized ✅")

    def estimate_distance(self, bbox, obj_class):
        """Simple distance estimator based on bounding box size"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        area = height * width
        
        if obj_class == "person":
            if height > 400: return "0.8m", 0.8
            elif height > 280: return "1.5m", 1.5
            elif height > 200: return "2.5m", 2.5
            else: return "6.0m", 6.0
        
        # Generic estimation
        if area > 80000: return "1.5m", 1.5
        elif area > 40000: return "3.0m", 3.0
        elif area > 20000: return "5.0m", 5.0
        else: return "12.0m", 12.0

    def is_object_straight_ahead(self, bbox, frame_width):
        x1, x2 = bbox[0], bbox[2]
        center_x = (x1 + x2) / 2
        return frame_width * 0.3 <= center_x <= frame_width * 0.7

    def speak_simple(self, text):
        if not self.engine or not text: return False
        try:
            self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except:
            return False

    def create_red_object_announcement(self, detections, frame_width):
        red_announcements = []
        for det in detections:
            obj_name = det['name']
            bbox = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
            distance_text, distance_meters = self.estimate_distance(bbox, obj_name)
            if distance_meters < 1.0 and self.is_object_straight_ahead(bbox, frame_width):
                red_announcements.append(f"Warning! {obj_name} obstacle straight ahead at {distance_text}")
        return ". ".join(red_announcements[:2]) if red_announcements else None

    def run(self):
        print("Smart Goggles Headless Mode Active")
        try:
            while True:
                # Capture frame
                frame_file = capture_frame("frame.jpg", width=640, height=480)
                
                # Detect objects
                results = self.yolo_model(frame_file)
                detections = results.pandas().xyxy[0]
                detections = detections[detections['confidence'] >= self.confidence_threshold]

                frame_width = 640  # Since we captured at 640x480

                # Announce very close red objects
                if self.frame_count % self.speak_delay == 0:
                    announcement = self.create_red_object_announcement(detections.to_dict('records'), frame_width)
                    if announcement:
                        print(f"Frame {self.frame_count}: {announcement}")
                        self.speak_simple(announcement)
                
                # Save annotated image
                results.save()  # saved in runs/detect/exp/
                print(f"Frame {self.frame_count} processed and saved ✅")
                
                self.frame_count += 1
                time.sleep(0.5)  # adjust capture interval
                
        except KeyboardInterrupt:
            print("Stopping headless detection...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.engine: self.engine.stop()
            print("Smart Goggles stopped")


def main():
    print("SMART GOGGLES - RED OBJECT ALERT MODE (HEADLESS)")
    goggles = FixedSmartGogglesHeadless()
    goggles.run()


if __name__ == "__main__":
    main()
