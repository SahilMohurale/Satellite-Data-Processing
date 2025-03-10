import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from sklearn.cluster import KMeans
from io import BytesIO
from zipfile import ZipFile
import tempfile
import os

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

if 'raster_height' not in st.session_state:
    st.session_state.raster_height = None

if 'raster_width' not in st.session_state:
    st.session_state.raster_width = None

if 'raster_crs' not in st.session_state:
    st.session_state.raster_crs = None

if 'raster_transform' not in st.session_state:
    st.session_state.raster_transform = None

if 'raster_data' not in st.session_state:
    st.session_state.raster_data = None


# Function to read and return band data
def read_band(band_file):
    with rasterio.open(band_file) as src:
        band = src.read(1).astype(np.float32)
    return band


# Function to calculate BAI
def calculate_BAI(red_band, swir_band):
    BAI = 1.0 * (swir_band - red_band) / (swir_band + red_band)
    return BAI


# Function to convert .tif tiles to .jpg
def convert_to_jpg(tif_dir, jpg_dir):
    os.makedirs(jpg_dir, exist_ok=True)
    for tif_file in os.listdir(tif_dir):
        if tif_file.endswith(".tif"):
            with rasterio.open(os.path.join(tif_dir, tif_file)) as src:
                image = src.read(1)  # Read the first band
                plt.imsave(os.path.join(jpg_dir, f"{os.path.splitext(tif_file)[0]}.jpg"), image, cmap='gray')


# Function to create zip file
def create_zip(folder_path, zip_name):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
    return zip_buffer.getvalue()


# Step 1: Upload a .tif file
st.title("Satellite Image Clustering and Tiling")

if st.session_state.step == 1:
    st.header("Step 1: Upload a .tif file")
    uploaded_tif = st.file_uploader("Upload a .tif file", type=['tif'])

    if uploaded_tif:
        with tempfile.NamedTemporaryFile(delete=False) as temp_tif:
            temp_tif.write(uploaded_tif.read())
            st.session_state.temp_tif_path = temp_tif.name

        with rasterio.open(st.session_state.temp_tif_path) as raster:
            st.session_state.raster_height = raster.height
            st.session_state.raster_width = raster.width
            st.session_state.raster_crs = raster.crs
            st.session_state.raster_transform = raster.transform
            st.session_state.raster_data = raster.read()

        st.session_state.step = 2
        st.experimental_rerun()

if st.session_state.step == 2:
    st.subheader("Uploaded .tif File")
    raster_data = st.session_state.raster_data
    fig, ax = plt.subplots()
    show(raster_data[0], ax=ax, title="Uploaded .tif File")
    st.pyplot(fig)

    # Step 2: Ask for the number of clusters and perform clustering
    st.header("Step 2: Perform Clustering")
    num_clusters = st.number_input("Enter the number of clusters:", min_value=2, max_value=20, value=5,
                                   key="num_clusters")

    if st.button("Perform Clustering", key="perform_clustering"):
        data = raster_data.reshape((raster_data.shape[0], -1)).T  # Reshape for clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
        clusters = kmeans.labels_.reshape((st.session_state.raster_height, st.session_state.raster_width))
        st.session_state.clusters = clusters
        st.session_state.step = 3
        st.experimental_rerun()

if st.session_state.step == 3:
    clusters = st.session_state.clusters
    st.subheader("Clustering Result")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(clusters, cmap='tab20b')
    plt.title("Clustering Result")
    st.pyplot(fig)

    # Step 3: Select a cluster to create a mask
    st.header("Step 3: Select a Cluster for Mask")
    selected_cluster = st.selectbox("Select a cluster:", range(np.max(clusters) + 1))

    if st.button("Create Mask", key="create_mask"):
        st.session_state.selected_cluster = selected_cluster
        mask = (clusters == selected_cluster).astype(np.uint8)
        st.session_state.mask = mask
        st.session_state.step = 4
        st.experimental_rerun()

if st.session_state.step == 4:
    mask = st.session_state.mask
    st.subheader(f"Selected Cluster Mask for Cluster {st.session_state.selected_cluster}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap='gray')
    plt.title(f"Mask for Cluster {st.session_state.selected_cluster}")
    st.pyplot(fig)

    # Step 4: Ask for tile dimensions and perform tiling
    st.header("Step 4: Tiling")
    tile_height = st.number_input("Enter tile height:", min_value=1, value=256, key="tile_height")
    tile_width = st.number_input("Enter tile width:", min_value=1, value=256, key="tile_width")

    if st.button("Perform Tiling", key="perform_tiling"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Tiling the raster
            raster_tiles_dir = os.path.join(temp_dir, "raster_tiles")
            mask_tiles_dir = os.path.join(temp_dir, "mask_tiles")
            os.makedirs(raster_tiles_dir, exist_ok=True)
            os.makedirs(mask_tiles_dir, exist_ok=True)

            tile_index = 0
            for i in range(0, st.session_state.raster_height, tile_height):
                for j in range(0, st.session_state.raster_width, tile_width):
                    window = rasterio.windows.Window(j, i, tile_width, tile_height)
                    transform = rasterio.windows.transform(window, st.session_state.raster_transform)
                    raster_tile = st.session_state.raster_data[:, i:i + tile_height, j:j + tile_width]
                    mask_tile = mask[i:i + tile_height, j:j + tile_width]

                    raster_tile_path = os.path.join(raster_tiles_dir, f"tile_{tile_index}.tif")
                    mask_tile_path = os.path.join(mask_tiles_dir, f"tile_{tile_index}.tif")

                    # Save raster tile
                    with rasterio.open(
                            raster_tile_path, 'w',
                            driver='GTiff',
                            height=raster_tile.shape[1],
                            width=raster_tile.shape[2],
                            count=raster_tile.shape[0],
                            dtype=raster_tile.dtype,
                            crs=st.session_state.raster_crs,
                            transform=transform) as dst:
                        dst.write(raster_tile)

                    # Save mask tile
                    with rasterio.open(
                            mask_tile_path, 'w',
                            driver='GTiff',
                            height=mask_tile.shape[0],
                            width=mask_tile.shape[1],
                            count=1,
                            dtype=mask_tile.dtype,
                            crs=st.session_state.raster_crs,
                            transform=transform) as dst:
                        dst.write(mask_tile, 1)

                    tile_index += 1

            # Convert tiles to .jpg
            raster_jpg_dir = os.path.join(temp_dir, "raster_jpgs")
            mask_jpg_dir = os.path.join(temp_dir, "mask_jpgs")
            convert_to_jpg(raster_tiles_dir, raster_jpg_dir)
            convert_to_jpg(mask_tiles_dir, mask_jpg_dir)

            # Create zip files for download
            raster_jpg_zip = create_zip(raster_jpg_dir, "raster_jpgs.zip")
            mask_jpg_zip = create_zip(mask_jpg_dir, "mask_jpgs.zip")

            st.session_state.raster_jpg_zip = raster_jpg_zip
            st.session_state.mask_jpg_zip = mask_jpg_zip
            st.session_state.step = 5
            st.experimental_rerun()

if st.session_state.step == 5:
    st.subheader("Download Tiled Images")
    st.download_button(
        label="Download Raster Tiles (JPG)",
        data=st.session_state.raster_jpg_zip,
        file_name="raster_jpgs.zip",
        mime="application/zip"
    )

    st.download_button(
        label="Download Mask Tiles (JPG)",
        data=st.session_state.mask_jpg_zip,
        file_name="mask_jpgs.zip",
        mime="application/zip"
    )
