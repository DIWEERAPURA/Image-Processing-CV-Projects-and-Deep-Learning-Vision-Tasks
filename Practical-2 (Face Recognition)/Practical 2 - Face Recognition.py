# Practical 2 - Face Recognition
# This script detects and compares faces between two images using the face_recognition library.
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np


def display_image(image, title='Image'):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def compare_and_mark_faces(main_image_path, target_image_path):
    # Load images using face_recognition (which uses numpy arrays)
    main_image = face_recognition.load_image_file(main_image_path)
    target_image = face_recognition.load_image_file(target_image_path)

    # Detect face locations and compute face encodings using a CNN model for accuracy
    main_face_locations = face_recognition.face_locations(main_image, model='cnn')
    main_face_encodings = face_recognition.face_encodings(main_image, known_face_locations=main_face_locations)

    target_face_locations = face_recognition.face_locations(target_image, model='cnn')
    target_face_encodings = face_recognition.face_encodings(target_image, known_face_locations=target_face_locations)

    if len(main_face_encodings) == 0:
        print('No faces found in the main image.')
        return
    reference_encoding = main_face_encodings[0]  # Assume the first detected face is the reference

    # Create a copy of the target image for marking
    marked_image = np.copy(target_image)
    for (top, right, bottom, left), face_encoding in zip(target_face_locations, target_face_encodings):
        matches = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.6)
        # Green box for a match, red for non-match
        color = (0, 255, 0) if matches[0] else (0, 0, 255)
        cv2.rectangle(marked_image, (left, top), (right, bottom), color, 2)

    display_image(marked_image, 'Face Recognition Result')


if __name__ == '__main__':
    main_image_path = 'main_face.jpg'  # Replace with the path to your main image
    target_image_path = 'target_faces.jpg'  # Replace with the path to your target image
    compare_and_mark_faces(main_image_path, target_image_path)
