# Practical 6 - Transfer Learning using VGG16 for Multi-Class Classification
# This script demonstrates transfer learning by adding custom layers to VGG16 for classifying images into multiple classes.
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the pre-trained VGG16 model (excluding top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for our classification task
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Change 3 to the number of classes in your dataset
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Data generators (ensure your dataset is organized in data/train and data/val folders)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(224,224),
                                                    batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('data/val', target_size=(224,224),
                                                        batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // 32,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // 32,
                    epochs=10)

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()
