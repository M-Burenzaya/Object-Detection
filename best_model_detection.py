from ultralytics import YOLO
import cv2
import math

# Load your trained OBB model
model = YOLO('./best.pt')

# Open video
cap = cv2.VideoCapture('data/test_videos/20250428_083245.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_083302.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_083322.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_083356.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_083416.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_185138.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_190556.mp4')
# cap = cv2.VideoCapture('data/test_videos/20250428_190611.mp4')

cv2.namedWindow("YOLOv8 OBB Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame, conf=0.7, verbose=False)

    # Get detections
    r = results[0]
    annotated_frame = r.plot()

    # Draw angle labels
    for xywhr in r.obb.xywhr:  # center_x, center_y, width, height, rotation_angle
        cx, cy, w, h, angle_rad = xywhr.tolist()
        angle_deg = math.degrees(angle_rad)
        text = f"{angle_deg:.1f}°"
        org = (int(cx), int(cy))
        cv2.putText(annotated_frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("YOLOv8 OBB Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
