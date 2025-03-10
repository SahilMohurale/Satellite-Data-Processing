import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

# Load the trained model with custom objects if needed
model = tf.keras.models.load_model('unet_model.h5', compile=False)


# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(target_size)  # Resize image to target size
    img = np.array(img) / 255.0  # Normalize image to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to load the original image (color)
def load_original_image(image_path, target_size=(1024, 1024)):
    img = cv2.imread(image_path)  # Read image using OpenCV (BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)  # Resize image to target size
    return img


# Predict and visualize with overlay
def predict_and_visualize(image_path):
    # Load and preprocess the input image
    preprocessed_image = load_and_preprocess_image(image_path)

    # Load the original image for visualization
    original_image = load_original_image(image_path)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Post-process the prediction
    predicted_mask = np.squeeze(prediction)  # Remove batch and channel dimensions

    # Threshold the predicted mask (you may adjust the threshold as needed)
    threshold = 0.5
    predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)

    # Resize predicted mask to match original image dimensions
    predicted_mask_resized = cv2.resize(predicted_mask_binary, (original_image.shape[1], original_image.shape[0]))

    # Create overlay image with translucent mask
    overlay = original_image.copy()
    overlay[predicted_mask_resized == 1] = [0, 255, 0]  # Overlay color (green)

    # Blend the original image and overlay with transparency
    output = cv2.addWeighted(original_image, 0.5, overlay, 0.5, 0)

    # Display the result using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Predicted Overlay")

    plt.tight_layout()
    plt.show()


# Predict and visualize on a given .jpg image
image_path = 'C:/Users/sahil/PycharmProjects/ISRO/ML/tmp/img/35.jpg'
predict_and_visualize(image_path)
