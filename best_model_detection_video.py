from ultralytics import YOLO
import cv2
import math
from pathlib import Path

# Load your trained OBB model
model = YOLO('./best.pt')

# Define input and output directories
input_dir = Path('data/test_videos')
output_dir = Path('data/output_videos')
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through all .mp4 videos in the folder
for video_path in input_dir.glob('*.mp4'):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path.name}")
        continue

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Processing: {video_path.name} → {output_path.name}")
    cv2.namedWindow("YOLOv8 OBB Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(source=frame, conf=0.7, verbose=False)
        r = results[0]
        annotated_frame = r.plot()

        # Draw angle on each box
        for xywhr in r.obb.xywhr:
            cx, cy, w, h, angle_rad = xywhr.tolist()
            angle_deg = math.degrees(angle_rad)
            text = f"{angle_deg:.1f}°"
            org = (int(cx), int(cy))
            cv2.putText(annotated_frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Show and save
        cv2.imshow("YOLOv8 OBB Detection", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    print(f"Saved to: {output_path}")

cv2.destroyAllWindows()
