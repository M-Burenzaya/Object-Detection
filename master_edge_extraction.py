import cv2
import numpy as np
import os

# Paths
master_image_path = 'data/master_image/master_1_edit.jpg'  # Adjust if needed
master_edge_path = './master_edge.png'  # Save here

# Load master image
master_img = cv2.imread(master_image_path)

if master_img is None:
    raise FileNotFoundError(f"Master image not found at {master_image_path}")

# Convert to grayscale
gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Save edge image
os.makedirs(os.path.dirname(master_edge_path), exist_ok=True)
cv2.imwrite(master_edge_path, edges)

# Find contours in the master edge image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Optional: Draw detected contours for checking
contour_preview = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 1)

# Resize for display
scale_percent = 50
w = int(contour_preview.shape[1] * scale_percent / 100)
h = int(contour_preview.shape[0] * scale_percent / 100)
contour_preview_display = cv2.resize(contour_preview, (w, h))

# Show result
cv2.imshow('Master Edges and Contours', contour_preview_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Master edge model saved at {master_edge_path}")
