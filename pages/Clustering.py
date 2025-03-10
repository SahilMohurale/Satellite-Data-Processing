import os
import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Streamlit application
st.title("K-means Clustering for Raster Data")

# User input for clustering
st.subheader("Upload TIFF File and Set Number of Clusters")
raster_file = st.file_uploader("Upload a TIFF file", type=['tif', 'tiff'], key='clustering_raster')
n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3, step=1, key='num_clusters')


# Helper function to perform K-means clustering
def perform_kmeans_clustering(raster_path, n_clusters):
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band

        # Flatten the image array and remove NaNs for clustering
        image_2d = image.reshape(-1, 1)
        valid_mask = ~np.isnan(image_2d)
        image_2d_valid = image_2d[valid_mask].reshape(-1, 1)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(image_2d_valid)
        clustered = np.full(image_2d.shape, np.nan)
        clustered[valid_mask] = kmeans.labels_

        clustered_image = clustered.reshape(image.shape)

    return clustered_image


# Main processing
if raster_file and n_clusters:
    try:
        # Save uploaded file to a temporary path
        raster_path = f'temp_raster.{raster_file.name.split(".")[-1]}'
        with open(raster_path, 'wb') as f:
            f.write(raster_file.read())

        # Perform K-means clustering
        clustered_image = perform_kmeans_clustering(raster_path, n_clusters)

        # Display clustered image
        st.subheader("Clustered Image")
        plt.figure(figsize=(10, 10))
        plt.imshow(clustered_image, cmap='tab20')
        plt.colorbar()
        plt.title(f'K-means Clustering with {n_clusters} Clusters')
        st.pyplot(plt)

        # Provide selection to show specific cluster
        st.subheader("Select Cluster to Display")
        cluster_selection = st.selectbox("Select a Cluster", options=list(range(n_clusters)))

        # Filter and display the selected cluster
        filtered_cluster = np.where(clustered_image == cluster_selection, clustered_image, np.nan)
        plt.figure(figsize=(10, 10))
        plt.imshow(filtered_cluster, cmap='tab20')
        plt.colorbar()
        plt.title(f'Selected Cluster: {cluster_selection}')
        st.pyplot(plt)

        # Clean up temporary file
        os.remove(raster_path)
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.warning("Please upload a TIFF file and set the number of clusters.")
