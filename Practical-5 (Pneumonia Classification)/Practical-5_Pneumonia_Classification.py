# Practical 5 - Pneumonia Classification using VGG16
# This script uses transfer learning with VGG16 for binary classification (pneumonia vs. normal).
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the pre-trained VGG16 model without its top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = Flatten()(x)
predictions = Dense(2, activation='softmax')(x)  # Binary classification (2 classes)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Data augmentation and generators (ensure your dataset is organized in data/train and data/val folders)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(224,224),
                                                    batch_size=10, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('data/val', target_size=(224,224),
                                                        batch_size=10, class_mode='categorical')

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // 10,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // 10,
                    epochs=5)

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()
