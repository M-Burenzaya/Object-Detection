import cv2
import numpy as np

# Paths
master_edge_path = './master_edge.png'
sample_image_path = 'data/test_images/20250428_083537.jpg'  # Change to your sample

# Load master edge image
master_edge = cv2.imread(master_edge_path, cv2.IMREAD_GRAYSCALE)

cv2.imshow

if master_edge is None:
    raise FileNotFoundError(f"Master edge image not found at {master_edge_path}")

# Find contours in master edge
master_contours, _ = cv2.findContours(master_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
master_contour = max(master_contours, key=cv2.contourArea)

# Load sample image
sample_img = cv2.imread(sample_image_path)

if sample_img is None:
    raise FileNotFoundError(f"Sample image not found at {sample_image_path}")

# Preprocess sample image
gray_sample = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
blurred_sample = cv2.GaussianBlur(gray_sample, (5, 5), 0)
edges_sample = cv2.Canny(blurred_sample, 50, 150)

# Find contours in sample image
sample_contours, _ = cv2.findContours(edges_sample.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare
detection_count = 0
threshold = 2  # Shape similarity threshold

sample_img_display = sample_img.copy()

# First, draw all contours in blue color
for cnt in sample_contours:
    area = cv2.contourArea(cnt)
    if area < 1000:
        continue
    cv2.drawContours(sample_img_display, [cnt], -1, (255, 0, 0), 2)  # Blue color (BGR: 255,0,0)

# Check all contours
for cnt in sample_contours:
    area = cv2.contourArea(cnt)
    if area < 1000:  # Ignore very small contours
        continue

    score = cv2.matchShapes(master_contour, cnt, 1, 0.0)

    if score < threshold:
        detection_count += 1
        cv2.drawContours(sample_img_display, [cnt], -1, (0, 255, 0), 2)
        print(f"Match {detection_count}: Similarity score = {score:.6f}")

if detection_count == 0:
    print("No matching objects detected.")
else:
    print(f"Total matches found: {detection_count}")

# Resize for display
scale_percent = 50
w = int(sample_img_display.shape[1] * scale_percent / 100)
h = int(sample_img_display.shape[0] * scale_percent / 100)
sample_img_display = cv2.resize(sample_img_display, (w, h))

# Show result
cv2.imshow('Shape Matching Result', sample_img_display)
cv2.imwrite('Test3_sample.jpeg', sample_img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
