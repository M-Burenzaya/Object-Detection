import cv2
import numpy as np

# Load master image
master_img = cv2.imread('data/master_image/master_1_edit.jpg', cv2.IMREAD_GRAYSCALE)

if master_img is None:
    raise FileNotFoundError("Master image not found.")


# Load master image
master_img = cv2.imread('data/master_image/master_1_edit.jpg', cv2.IMREAD_GRAYSCALE)

if master_img is None:
    raise FileNotFoundError("Master image not found.")

# Resize master image
scale_percent = 50  # Try scaling down to 50% or 40%
width = int(master_img.shape[1] * scale_percent / 100)
height = int(master_img.shape[0] * scale_percent / 100)
master_img = cv2.resize(master_img, (width, height), interpolation=cv2.INTER_AREA)

w_template, h_template = master_img.shape[::-1]

# Open video source
cap = cv2.VideoCapture('data/test_videos/20250428_083416.mp4')

if not cap.isOpened():
    raise IOError("Cannot open video source")

# Template Matching settings
threshold = 0.6  # lowered threshold for video

def non_max_suppression_fast(boxes, scores, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: slightly blur to reduce noise
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    result = cv2.matchTemplate(gray_frame, master_img, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    detections = []
    for loc in locations:
        x, y = loc
        score = result[y, x]
        detections.append((x, y, score))

    detections = sorted(detections, key=lambda x: x[2], reverse=True)

    boxes = []
    scores = []

    for (x, y, score) in detections:
        boxes.append([x, y, x + w_template, y + h_template])
        scores.append(score)

    boxes = np.array(boxes)
    scores = np.array(scores)

    if len(boxes) > 0:
        picked_indices = non_max_suppression_fast(boxes, scores, overlapThresh=0.3)

        # Draw results
        for i in picked_indices:
            (x1, y1, x2, y2) = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show live frame
    cv2.imshow('Template Matching Video Stream', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
