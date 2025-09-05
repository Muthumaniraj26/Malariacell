import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset path (update this to your local path)
dataset_dir = r"C:/Users/muthumaniraj/Documents/Malaria cell/cell_images"

# Data preprocessing & augmentation
datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2)  # 20% for validation

train_data = datagen.flow_from_directory(dataset_dir,
                                         target_size=(64,64),
                                         batch_size=32,
                                         class_mode='binary',
                                         subset='training')

val_data = datagen.flow_from_directory(dataset_dir,
                                       target_size=(64,64),
                                       batch_size=32,
                                       class_mode='binary',
                                       subset='validation')

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_data,
                    epochs=10,
                    validation_data=val_data)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

# Evaluate
loss, acc = model.evaluate(val_data)
print(f"✅ Validation Accuracy: {acc:.2f}")
