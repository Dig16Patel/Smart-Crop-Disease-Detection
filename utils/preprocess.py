import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses the image for the model.
    1. Resizes to target_size.
    2. Converts to NumPy array.
    3. Normalizes pixel values to [0, 1].
    4. Expands dimensions to match model input shape (batch_size, height, width, channels).
    
    Args:
        image: PIL Image object.
        target_size: Tuple (height, width) for resizing.
        
    Returns:
        Preprocessed image array with shape (1, height, width, 3).
    """
    # 1. Resize
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    
    # 2. Convert to array
    image_array = np.array(image)
    
    # 3. Normalize pixel values
    image_array = image_array.astype("float32") / 255.0
    
    # 4. Expand dimensions (add batch dimension)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

if __name__ == "__main__":
    # Simple test when running this file directly
    print("Testing preprocessing function...")
    try:
        # Create a dummy image for testing (100x100 white image)
        dummy_image = Image.new('RGB', (100, 100), color = 'white')
        processed = preprocess_image(dummy_image)
        print(f"Original size: {dummy_image.size}")
        print(f"Processed shape: {processed.shape}")
        print(f"Min value: {processed.min()}, Max value: {processed.max()}")
        print("Test Passed! ✅")
    except Exception as e:
        print(f"Test Failed! ❌ Error: {e}")
