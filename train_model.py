import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import json

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 5
DATASET_DIR = "dataset/train"
MODEL_SAVE_PATH = "models/crop_disease_model.h5"
INDICES_SAVE_PATH = "models/class_indices.json"

# --- Step 1: Generate Dummy Data (If needed) ---
def generate_dummy_data():
    classes = ["Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Healthy"]
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    for class_name in classes:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Generating dummy images for {class_name}...")
            # Generate 10 random images per class
            for i in range(10):
                # Random noise image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))

print("Checking for dataset...")
# Check if dataset has images, if not generate them
if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
    generate_dummy_data()
else:
    # Check if subfolders exist
    if not any(os.path.isdir(os.path.join(DATASET_DIR, d)) for d in os.listdir(DATASET_DIR)):
        generate_dummy_data()

print("Dataset ready.")

# --- Step 2: Load Data ---
print("Loading data...")
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class indices for later use in prediction
class_indices = train_generator.class_indices
print(f"Classes found: {class_indices}")
with open(INDICES_SAVE_PATH, 'w') as f:
    json.dump(class_indices, f)
    print(f"Saved class indices to {INDICES_SAVE_PATH}")

# --- Step 3: Build Model ---
print("Building model...")
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 4: Train Model ---
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- Step 5: Save Model ---
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print("Training Complete! ✅")
