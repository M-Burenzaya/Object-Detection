from ultralytics import YOLO
import cv2
import math
from pathlib import Path

# Load your trained OBB model
model = YOLO('./best.pt')

# Define input and output directories
input_dir = Path('data/test_images')
output_dir = Path('data/output_images')
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through all .jpg and .png images in the folder
for img_path in input_dir.glob('*.[jp][pn]g'):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to load {img_path.name}")
        continue

    # Run inference
    results = model.predict(source=image, conf=0.7, verbose=False)
    r = results[0]
    annotated_image = r.plot()

    # Draw angle on each box
    for xywhr in r.obb.xywhr:
        cx, cy, w, h, angle_rad = xywhr.tolist()
        angle_deg = math.degrees(angle_rad)
        text = f"{angle_deg:.1f}°"
        org = (int(cx), int(cy))
        cv2.putText(annotated_image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Save output image
    output_path = output_dir / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Processed {img_path.name} → {output_path.name}")

print("All images processed.")
