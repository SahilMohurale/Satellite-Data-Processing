import os
import zipfile
import streamlit as st
import rasterio
import numpy as np
from PIL import Image
import shutil

# Streamlit application
st.title("TIFF Mask to JPEG Converter")

# User input for folder of TIFF files
st.subheader("Upload Folder Containing TIFF Mask Files")
uploaded_files = st.file_uploader("Upload TIFF mask files", type=['tif', 'tiff'], accept_multiple_files=True)


def convert_mask_tif_to_jpg(tif_file, output_folder):
    with rasterio.open(tif_file) as src:
        image = src.read(1)  # Read the first band
        image = (image * 255).astype(np.uint8)  # Scale values from 0-1 to 0-255
        jpg_image = Image.fromarray(image)

        jpg_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(tif_file))[0] + '.jpg')
        jpg_image.save(jpg_filename)
    return jpg_filename


def create_zip_of_jpg_files(jpg_files, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for file in jpg_files:
            zipf.write(file, os.path.basename(file))


if uploaded_files:
    output_folder = "converted_jpgs"
    os.makedirs(output_folder, exist_ok=True)

    # Convert each TIFF file to JPEG
    jpg_files = []
    for tif_file in uploaded_files:
        temp_tif_path = os.path.join(output_folder, tif_file.name)
        with open(temp_tif_path, 'wb') as f:
            f.write(tif_file.getbuffer())
        jpg_path = convert_mask_tif_to_jpg(temp_tif_path, output_folder)
        jpg_files.append(jpg_path)

    # Create a ZIP file of all JPEG files
    output_zip_path = os.path.join(output_folder, "converted_jpgs.zip")
    create_zip_of_jpg_files(jpg_files, output_zip_path)

    # Provide download button for the ZIP file
    st.subheader("Download Converted JPEG Files")
    with open(output_zip_path, "rb") as f:
        st.download_button("Download ZIP", f, file_name="converted_jpgs.zip")

    # Clean up temporary files
    shutil.rmtree(output_folder)
else:
    st.warning("Please upload a folder containing TIFF files.")
