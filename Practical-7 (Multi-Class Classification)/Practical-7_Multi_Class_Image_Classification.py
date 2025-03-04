# Practical 7 - Multi-Class Image Classification
# This script demonstrates two approaches to classify images from the CIFAR-10 dataset:
# 1. A simple Artificial Neural Network (ANN)
# 2. A Convolutional Neural Network (CNN)
# The script loads the dataset, normalizes the images, trains both models, and then evaluates them.

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# Normalize images to the range [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert labels to one-hot encoding for training
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# -------------------------------
# Build and Train a Simple ANN Model
# -------------------------------
def build_ann_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

ann_model = build_ann_model()
history_ann = ann_model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
ann_loss, ann_acc = ann_model.evaluate(x_test, y_test_cat, verbose=0)
print("ANN Test Accuracy:", ann_acc)

y_pred_ann = np.argmax(ann_model.predict(x_test), axis=1)
print("Classification Report for ANN:")
print(classification_report(y_test, y_pred_ann))

# -------------------------------
# Build and Train a CNN Model
# -------------------------------
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()
history_cnn = cnn_model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test_cat, verbose=0)
print("CNN Test Accuracy:", cnn_acc)

y_pred_cnn = np.argmax(cnn_model.predict(x_test), axis=1)
print("Classification Report for CNN:")
print(classification_report(y_test, y_pred_cnn))

# -------------------------------
# Plot CNN Training History as an example
# -------------------------------
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
