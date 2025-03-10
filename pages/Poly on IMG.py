import streamlit as st
from PIL import Image, ImageDraw
import os
import numpy as np

# Streamlit application
st.title("Polygon Label Visualizer")

# User input for image and label files
st.subheader("Upload Image and Label Files")
uploaded_image = st.file_uploader("Upload JPEG image", type=['jpg', 'jpeg'])
uploaded_label_file = st.file_uploader("Upload Label File", type=['txt'])


def draw_polygon(image_path, label_coordinates):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    coords = np.array(label_coordinates.split()).astype(float)

    # Ensure the number of coordinates is even
    if len(coords) % 2 != 0:
        st.error("The number of coordinates is not even. Please check your label file.")
        return image

    coords = coords.reshape(-1, 2)
    width, height = image.size
    polygon = [(x * width, y * height) for x, y in coords]

    draw.polygon(polygon, outline="red")
    return image


if uploaded_image and uploaded_label_file:
    # Save uploaded image temporarily
    image_path = os.path.join("temp_image.jpg")
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())

    # Read label coordinates from the uploaded text file
    label_coordinates = uploaded_label_file.read().decode("utf-8").strip()

    # Draw polygon on the image
    labeled_image = draw_polygon(image_path, label_coordinates)

    # Display the image
    st.image(labeled_image, caption="Labeled Image", use_column_width=True)

    # Clean up temporary file
    os.remove(image_path)
else:
    st.warning("Please upload both an image and label file.")
