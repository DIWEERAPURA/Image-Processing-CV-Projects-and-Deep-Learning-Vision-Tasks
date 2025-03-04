# Practical 1 - Basic Image Processing
# This script performs image preprocessing tasks using OpenCV.

import cv2
import matplotlib.pyplot as plt

# Load Image
image_path = 'example.jpg'  # Update this with your image path
image = cv2.imread(image_path)
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

# Apply Gaussian Blurring
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)

# Edge Detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Thresholding
_, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)

# Contour Detection
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
