import cv2
import pyttsx3
import torch
from time import sleep

# Load YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # 'n' = nano version (small & fast)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speech speed

# Start webcam
cap = cv2.VideoCapture(0)

# Only speak every N frames
speak_delay = 30
frame_count = 0 

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Inference
        results = model(frame)
        detections = results.pandas().xyxy[0]

        frame_count += 1

        # Draw boxes and speak only every 30 frames (~1 sec)
        if frame_count % speak_delay == 0:
            if len(detections) > 0:
                object_names = detections['name'].unique().tolist()
                objects_str = ", ".join(object_names)
                print("Detected:", objects_str)

                # Speak
                engine.say(f"Obstacle detected: {objects_str} ahead")
                engine.runAndWait()
            else:
                print("No obstacles")

        # Display
        results.render()
        cv2.imshow('Smart Goggles AI View', results.ims[0])

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

