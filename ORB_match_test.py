import cv2
import numpy as np

# Load images
master_img = cv2.imread('data/master_image/master_1_edit.jpg', cv2.IMREAD_GRAYSCALE)
sample_img = cv2.imread('data/test_images/20250428_083537.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create(5000)

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(master_img, None)
kp2, des2 = orb.detectAndCompute(sample_img, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches (for visualization)
matched_img = cv2.drawMatches(master_img, kp1, sample_img, kp2, matches[:50], None, flags=2)

# Homography estimation
if len(matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is not None:
        # Get tilt angle from homography
        angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
        print(f"Tilt Angle: {angle:.2f} degrees")
        
        # Extract master object edges using Canny
        edges = cv2.Canny(master_img, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]  # Pick the biggest
        main_contour = main_contour.reshape(-1, 1, 2).astype(np.float32)
        transformed_contour = cv2.perspectiveTransform(main_contour, M)
        sample_img_color = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
        sample_img_color = cv2.polylines(sample_img_color, [np.int32(transformed_contour)], True, (0, 255, 0), 2, cv2.LINE_AA)

        scale_percent = 30

        w = int(sample_img_color.shape[1] * scale_percent / 100)
        h = int(sample_img_color.shape[0] * scale_percent / 100)
        sample_img_display = cv2.resize(sample_img_color, (w, h))

        w = int(matched_img.shape[1] * scale_percent / 100)
        h = int(matched_img.shape[0] * scale_percent / 100)
        matched_img_display = cv2.resize(matched_img, (w, h))

        # Show results
        cv2.imshow('Detection', sample_img_display)
        cv2.imshow('Matches', matched_img_display)

        # cv2.imwrite('Test2_sample.jpeg', sample_img_color)
        cv2.imwrite('Test2_match.jpeg', matched_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Homography could not be computed.")
else:
    print("Not enough matches found.")
