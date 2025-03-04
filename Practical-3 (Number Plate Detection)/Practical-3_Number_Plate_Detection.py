# Practical 3 - Number Plate Detection with OCR
# This script detects potential number plate regions and performs OCR using EasyOCR.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr


def display_image(image, title='Image'):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def find_number_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 450)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50000 < area < 100000:  # Adjust these thresholds based on your image size
            potential_plates.append(cnt)
    plate_img = image.copy()
    cv2.drawContours(plate_img, potential_plates, -1, (0, 255, 0), 2)
    return image, plate_img, potential_plates


def perform_ocr_on_plate(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    plate_region = image[y:y + h, x:x + w]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate_region)
    return result, plate_region


def number_plate_detection(image_path):
    original, marked, plates = find_number_plate(image_path)
    display_image(marked, 'Detected Plate Regions')

    if not plates:
        print("No potential number plates detected.")
        return

    for cnt in plates:
        result, plate_region = perform_ocr_on_plate(original, cnt)
        print("OCR Result:", result)
        display_image(plate_region, 'Plate Region for OCR')


if __name__ == '__main__':
    image_path = 'vehicle.jpg'  # Update with the path to your vehicle image
    number_plate_detection(image_path)
