import cv2
from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
from plyer import notification

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# -----------------------------
# Variables
# -----------------------------
slouch_start = None
last_alert = 0
ALERT_COOLDOWN = 30
log_data = []

# Calibration variables
calibration_start = time.time()
calibration_values = []
CALIBRATION_TIME = 5
baseline_distance = None


# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    if results[0].keypoints is not None:

        kpts = results[0].keypoints.xy[0]

        ear = kpts[3]
        shoulder = kpts[5]
        hip = kpts[11]

        # Distance calculation
        ear_shoulder_dist = np.linalg.norm(ear - shoulder)
        shoulder_hip_dist = np.linalg.norm(shoulder - hip)

        if shoulder_hip_dist != 0:
            normalized_distance = ear_shoulder_dist / shoulder_hip_dist
        else:
            normalized_distance = 0

        cv2.putText(annotated, f"Norm Dist: {normalized_distance:.2f}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # -----------------------------
        # Calibration Phase
        # -----------------------------
        if baseline_distance is None:

            if time.time() - calibration_start < CALIBRATION_TIME:
                calibration_values.append(normalized_distance)

                cv2.putText(annotated, "Calibrating... Sit Straight",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

            else:
                baseline_distance = np.mean(calibration_values)
                print("Baseline Distance:", baseline_distance)

        # -----------------------------
        # After Calibration
        # -----------------------------
        else:

            # Create severity levels relative to baseline
            mild_threshold = baseline_distance * 0.92
            moderate_threshold = baseline_distance * 0.83
            severe_threshold = baseline_distance * 0.74

            if normalized_distance >= mild_threshold:
                posture_status = "Good"
                color = (0, 255, 0)

            elif normalized_distance >= moderate_threshold:
                posture_status = "Mild Slouch"
                color = (0, 255, 255)

            elif normalized_distance >= severe_threshold:
                posture_status = "Moderate Slouch"
                color = (0, 140, 255)

            else:
                posture_status = "Severe Slouch"
                color = (0, 0, 255)

            cv2.putText(annotated, posture_status,
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

            # Alert only for Severe
            if posture_status == "Severe Slouch":

                if slouch_start is None:
                    slouch_start = time.time()

                slouch_duration = time.time() - slouch_start

                cv2.putText(annotated, f"Severe: {int(slouch_duration)}s",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

                if slouch_duration > 30:
                    if time.time() - last_alert > ALERT_COOLDOWN:
                        notification.notify(
                            title="Posture Alert",
                            message="Severe Slouch Detected! Sit Straight!",
                            timeout=5
                        )
                        last_alert = time.time()

            else:
                slouch_start = None

            # Log data
            log_data.append({
                "time": time.time(),
                "normalized_distance": float(normalized_distance),
                "baseline": float(baseline_distance),
                "status": posture_status
            })

    cv2.imshow("Adaptive Posture Monitor", annotated)

    if cv2.waitKey(1) == ord('q'):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(log_data)
df.to_csv("posture_log.csv", index=False)

print("Posture data saved to posture_log.csv")