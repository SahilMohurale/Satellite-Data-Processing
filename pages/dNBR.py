import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Streamlit application
st.title("Burned Area Index (BAI) Difference Calculator")

# User input for band files
st.subheader("Upload Band Files")
pre_fire_files = {
    'red_pre': st.file_uploader("Upload Pre-fire Red Band", type=['tif'], key='red_pre'),
    'swir_pre': st.file_uploader("Upload Pre-fire SWIR Band", type=['tif'], key='swir_pre')
}

post_fire_files = {
    'red_post': st.file_uploader("Upload Post-fire Red Band", type=['tif'], key='red_post'),
    'swir_post': st.file_uploader("Upload Post-fire SWIR Band", type=['tif'], key='swir_post')
}

def read_band(band_file):
    try:
        with rasterio.open(band_file) as src:
            band = src.read(1).astype(np.float32)
        return band
    except Exception as e:
        st.error(f"Error reading band file: {e}")
        return None

def calculate_BAI(red_band, swir_band):
    BAI = 1.0 * (swir_band - red_band) / (swir_band + red_band)
    return BAI

if all(pre_fire_files.values()) and all(post_fire_files.values()):  # Ensure all files are uploaded
    pre_fire_bands = {}
    post_fire_bands = {}

    for band, uploaded_file in pre_fire_files.items():
        if uploaded_file is not None:
            pre_fire_bands[band] = read_band(BytesIO(uploaded_file.read()))

    for band, uploaded_file in post_fire_files.items():
        if uploaded_file is not None:
            post_fire_bands[band] = read_band(BytesIO(uploaded_file.read()))

    if pre_fire_bands and post_fire_bands:  # Ensure bands are read successfully
        pre_fire_red = pre_fire_bands['red_pre']
        pre_fire_swir = pre_fire_bands['swir_pre']
        post_fire_red = post_fire_bands['red_post']
        post_fire_swir = post_fire_bands['swir_post']

        # Calculate BAI for pre-fire and post-fire images
        pre_fire_BAI = calculate_BAI(pre_fire_red, pre_fire_swir)
        post_fire_BAI = calculate_BAI(post_fire_red, post_fire_swir)

        # Display difference in BAI
        BAI_difference = post_fire_BAI - pre_fire_BAI

        st.subheader("BAI Difference Map")
        plt.imshow(BAI_difference, cmap='RdYlGn')
        plt.colorbar(label='BAI Difference')
        st.pyplot()

        # Provide option to download BAI difference map
        st.download_button(
            label="Download BAI Difference Map as TIFF",
            data=BytesIO(BAI_difference.tobytes()),
            file_name="BAI_Difference_Map.tif",
            mime="image/tiff"
        )
    else:
        st.error("Failed to read band files. Please check the uploaded files.")
else:
    st.warning("Please upload all pre-fire and post-fire band files.")
