import cv2
from ultralytics import YOLO
import time
import threading

# Load the model
# NOTE: For Fire and Smoke detection, you must train the model as per instructions.
# For now, we use the standard yolov8n.pt which detects 'person' (Class 0).
# Once trained, replace "yolov8n.pt" with "runs/detect/train/weights/best.pt"
try:
    model = YOLO("runs/detect/train/weights/best.pt")
    print("Loaded custom Fire/Smoke model.")
except:
    print("Custom model not found. Loading standard YOLOv8n (Human detection only).")
    print("To detect Fire/Smoke, please complete the training step.")
    model = YOLO("yolov8n.pt")

# Camera URL
CAMERA_URL = "http://100.123.231.128:8080/video"

def run_surveillance():
    print(f"Connecting to camera: {CAMERA_URL}")
    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        print("Error: Could not open camera stream.")
        return

    # Full screen or large window
    window_name = "BIOXEN Surveillance - Human | Fire | Smoke"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed to read frame.")
            # Verify connection logic or retry
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        # Run inference
        results = model(frame, stream=True)

        critical_detected = False
        detected_types = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Class 0 = Person (Standard model)
                # If trained custom model: 0=Person, 1=Fire, 2=Smoke
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = ""
                color = (0, 255, 0) # Green default

                # Logic for labels
                if cls == 0:
                    label = "Human"
                    color = (0, 255, 0) # Green
                elif cls == 1:
                    label = "Fire"
                    color = (0, 0, 255) # Red
                    critical_detected = True
                    detected_types.append("Fire")
                elif cls == 2:
                    label = "Smoke"
                    color = (0, 165, 255) # Orange
                    critical_detected = True
                    detected_types.append("Smoke")
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                t_size = cv2.getTextSize(f"{label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                
                # Text
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Alert Logic (Simulated integration)
        if critical_detected:
            alert_msg = f"CRITICAL: {', '.join(set(detected_types))} DETECTED!"
            cv2.putText(frame, alert_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Here you would trigger the WebSocket or HTTP Call to Backend
            # send_alert_to_backend(detected_types)

        cv2.imshow(window_name, frame)

        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_surveillance()
