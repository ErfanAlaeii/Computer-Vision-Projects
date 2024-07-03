import cv2
import numpy as np
import argparse
import os

# Using argparse to parse the image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Load image
origImg = cv2.imread(args["image"])

# Check if image is loaded successfully
if origImg is None:
    print(f"Error: Unable to open image file {args['image']}")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
eql = cv2.equalizeHist(gray)

blur = cv2.medianBlur(eql, 9)

# Apply adaptive thresholding for better performance on varying lighting conditions
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 20)

# Apply morphological operations to close gaps and remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

# morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

# Apply edge detection
edges = cv2.Canny(morphed, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create directory for saving IC images
os.makedirs("IC", exist_ok=True)

IC_No = 1
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) >= 4:
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h
        # Check for aspect ratio and size to filter out non-IC objects
        if 1 < aspect_ratio < 1.5 and 350 < area < 10050:
            cv2.putText(origImg, 'IC', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(origImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            IC = origImg[y:y + h, x:x + w]
            cv2.imwrite(f"IC/IC-{IC_No}.jpg", IC)
            IC_No += 1

# Show the final image with detected ICs
cv2.imshow("Detected ICs", origImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
