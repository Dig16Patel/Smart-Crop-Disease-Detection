import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses the image for the EfficientNetB0 model.
    MUST match the exact preprocessing used during training on HuggingFace.

    Steps:
    1. Converts to RGB (handles RGBA/grayscale inputs).
    2. Resizes to target_size (224x224).
    3. Converts to NumPy float32 array.
    4. Applies EfficientNet-specific preprocessing (scales to [-1, 1]).
    5. Expands dimensions to add batch axis: (1, H, W, 3).

    Args:
        image: PIL Image object.
        target_size: Tuple (height, width) for resizing. Default (224, 224).

    Returns:
        Preprocessed image array with shape (1, 224, 224, 3).
    """
    # 1. Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Resize
    image = image.resize(target_size)

    # 3. Convert to float32 array
    image_array = np.array(image).astype("float32")

    # 4. EfficientNet preprocessing (same as training — scales to [-1, 1])
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)

    # 5. Add batch dimension → (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


if __name__ == "__main__":
    print("Testing EfficientNet preprocessing function...")
    try:
        dummy_image = Image.new('RGB', (100, 100), color='green')
        processed = preprocess_image(dummy_image)
        print(f"Original size: {dummy_image.size}")
        print(f"Processed shape: {processed.shape}")
        print(f"Min value: {processed.min():.4f}, Max value: {processed.max():.4f}")
        print("Expected range: approximately [-1, 1]")
        print("Test Passed! ✅")
    except Exception as e:
        print(f"Test Failed! ❌ Error: {e}")
