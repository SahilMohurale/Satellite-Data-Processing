import os
import zipfile
import tempfile
import numpy as np
import rasterio
from rasterio.windows import Window
import streamlit as st
from io import BytesIO

# Streamlit application
st.title("Raster Tiling Tool")

# User input for tiling
st.subheader("Upload Files for Tiling")
raster_file = st.file_uploader("Upload a VRT or TIFF file (max 2GB)", type=['vrt', 'tif'], key='tiling_raster')
tile_size = st.text_input("Enter Tile Size (e.g., 256x256)", value="256x256")


def save_uploaded_file(uploaded_file, file_path):
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())


def create_tiles(raster_path, tile_size):
    try:
        with rasterio.open(raster_path) as src:
            tile_width, tile_height = tile_size
            n_cols = src.width // tile_width + (1 if src.width % tile_width else 0)
            n_rows = src.height // tile_height + (1 if src.height % tile_height else 0)

            temp_dir = tempfile.mkdtemp()
            tile_paths = []

            for i in range(n_rows):
                for j in range(n_cols):
                    window = Window(j * tile_width, i * tile_height, tile_width, tile_height)
                    out_path = os.path.join(temp_dir, f"tile_{i}_{j}.tif")
                    with rasterio.open(
                            out_path, 'w',
                            driver='GTiff',
                            height=tile_height,
                            width=tile_width,
                            count=src.count,
                            dtype=src.dtypes[0],
                            crs=src.crs,
                            transform=rasterio.windows.transform(window, src.transform)
                    ) as dst:
                        dst.write(src.read(window=window))
                    tile_paths.append(out_path)

        return tile_paths, temp_dir
    except Exception as e:
        st.error(f"Error creating tiles: {e}")
        return [], None


def create_zip_file(file_paths):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer


if raster_file and tile_size:
    try:
        # Check file size (Streamlit already limits uploads to below 2GB)
        if raster_file.size > 2 * 1024 * 1024 * 1024:
            st.error("File size exceeds 2GB limit. Please upload a smaller file.")
        else:
            # Save the uploaded raster file to a temporary path
            raster_path = f'temp_raster.{raster_file.name.split(".")[-1]}'
            save_uploaded_file(raster_file, raster_path)

            # Parse the tile size
            tile_width, tile_height = map(int, tile_size.split('x'))

            # Create tiles
            tile_paths, temp_dir = create_tiles(raster_path, (tile_width, tile_height))

            if tile_paths:
                st.success(f"Created {len(tile_paths)} tiles.")

                # Create a zip file for download
                zip_buffer = create_zip_file(tile_paths)

                # Provide download button
                st.download_button(
                    label="Download Tiles as ZIP",
                    data=zip_buffer,
                    file_name="tiles.zip",
                    mime="application/zip"
                )

                # Clean up the temporary directory
                for tile_path in tile_paths:
                    os.remove(tile_path)
                os.rmdir(temp_dir)
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.warning("Please upload a VRT or TIFF file and enter the tile size (e.g., 256x256).")
