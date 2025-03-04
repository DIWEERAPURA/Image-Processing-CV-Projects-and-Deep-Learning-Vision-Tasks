# Practical 4 - Live Object Detection using OpenCV DNN
# This script uses a pre-trained MobileNet SSD model to perform object detection on an image.
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_objects_in_image(image_path):
    # Load class labels from the COCO dataset (ensure coco.names is in your folder)
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Load the model configuration and weights (download these files beforehand)
    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    image = cv2.imread(image_path)
    classIds, confidences, bbox = net.detect(image, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bbox):
            label = classes[classId - 1]
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'{label}: {confidence * 100:.1f}%', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Object Detection')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image_path = 'test_image.jpg'  # Update with your test image path
    detect_objects_in_image(image_path)
