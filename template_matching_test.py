import cv2
import numpy as np

# Load master and sample images
master_img = cv2.imread('data/master_image/master_1_edit.jpg', cv2.IMREAD_GRAYSCALE)
sample_img = cv2.imread('data/test_images/20250428_083537.jpg', cv2.IMREAD_GRAYSCALE)

if master_img is None:
    raise FileNotFoundError("Master image not found.")
if sample_img is None:
    raise FileNotFoundError("Sample image not found.")

# Template Matching
result = cv2.matchTemplate(sample_img, master_img, cv2.TM_CCOEFF_NORMED)

# Matching threshold
threshold = 0.5  # Adjust carefully
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))  # (x, y) points

# Save all detections with their scores
detections = []

for loc in locations:
    x, y = loc
    score = result[y, x]
    detections.append((x, y, score))

# Sort detections by score (best first)
detections = sorted(detections, key=lambda x: x[2], reverse=True)

# Prepare for Non-Maximum Suppression (NMS)
boxes = []
scores = []

w, h = master_img.shape[::-1]  # Template size

for (x, y, score) in detections:
    boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2)
    scores.append(score)

boxes = np.array(boxes)
scores = np.array(scores)

# Apply NMS
def non_max_suppression_fast(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    # Coordinates
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]

    return keep

# Perform NMS
picked_indices = non_max_suppression_fast(boxes, scores, overlapThresh=0.3)

# Draw detections
sample_img_display = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)

for i in picked_indices:
    (x1, y1, x2, y2) = boxes[i]
    cv2.rectangle(sample_img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(f"Total matches after NMS: {len(picked_indices)}")

# Resize for display
scale_percent = 50
w_disp = int(sample_img_display.shape[1] * scale_percent / 100)
h_disp = int(sample_img_display.shape[0] * scale_percent / 100)
sample_img_display = cv2.resize(sample_img_display, (w_disp, h_disp))

# Show result
cv2.imshow('Template Matching with NMS', sample_img_display)
cv2.imwrite('Test4_sample.jpeg', sample_img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
