import os
from flask import Flask, render_template, request, send_file
import rasterio
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Function to read band from uploaded file
def read_band(file):
    try:
        # Save file to a temporary location
        temp_filename = os.path.join('D:\\tif_flask_app\\tmp', os.path.basename(file.filename))
        file.save(temp_filename)

        # Open saved file with rasterio
        with rasterio.open(temp_filename) as src:
            band = src.read(1).astype(np.float32)

        # Delete temporary file after use (optional)
        os.remove(temp_filename)

        return band
    except Exception as e:
        print(f"Error reading band file: {e}")
        return None


# Function to normalize band
def normalize_band(band):
    band_min, band_max = np.percentile(band, (2, 98))  # Ignore extreme values for better contrast
    band = np.clip((band - band_min) / (band_max - band_min), 0, 1)
    return (band * 255).astype(np.uint8)


# Function to create true color image from selected bands
def create_true_color_image(band_files, bands_selected):
    bands = []
    for band in bands_selected:
        band_data = read_band(band_files[band])
        if band_data is not None:
            bands.append(normalize_band(band_data))
        else:
            return None
    true_color_image = np.dstack(bands)
    return true_color_image


# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file uploads
        uploaded_files = {}
        for band in range(1, 4):  # Adjust range to match number of bands
            file = request.files.get(f'file{band}')
            if file and file.filename != '':
                uploaded_files[f'B{band}'] = file

        selected_bands = ['B1', 'B2', 'B3']  # Adjust band identifiers as per your form

        if len(selected_bands) == 3 and all(band in uploaded_files for band in selected_bands):
            # Process selected bands
            band_files = {band: uploaded_files[band] for band in selected_bands}
            true_color_image = create_true_color_image(band_files, selected_bands)

            if true_color_image is not None:
                # Save the image to a temporary file (PNG format)
                img = Image.fromarray(true_color_image)
                temp_filename_png = 'D:\\tif_flask_app\\tmp\\true_color_image.png'  # Adjust path as per your environment
                img.save(temp_filename_png)

                # Convert the image to TIFF format
                temp_filename_tif = 'D:\\tif_flask_app\\tmp\\true_color_image.tif'
                with rasterio.open(
                        temp_filename_tif, 'w',
                        driver='GTiff',
                        height=true_color_image.shape[0],
                        width=true_color_image.shape[1],
                        count=3,
                        dtype=true_color_image.dtype
                ) as dst:
                    for i in range(3):  # Write each band
                        dst.write(true_color_image[:, :, i], i + 1)

                # Return the rendered HTML with the image displayed and download links
                return render_template('result.html', image_png=temp_filename_png, image_tif=temp_filename_tif)
            else:
                return "Failed to create the true color image. Please check the uploaded files and ensure they are valid TIFF files."
        else:
            return "Please select exactly three bands and upload all required files."

    return render_template('index.html')


# Route to serve the image file (PNG format)
@app.route('/download_png')
def download_png():
    filename = 'D:\\tif_flask_app\\tmp\\true_color_image.png'  # Adjust path as per your environment
    return send_file(filename, as_attachment=True, attachment_filename='true_color_image.png')


# Route to serve the image file (TIFF format)
@app.route('/download_tif')
def download_tif():
    filename = 'D:\\tif_flask_app\\tmp\\true_color_image.tif'  # Adjust path as per your environment
    return send_file(filename, as_attachment=True, attachment_filename='true_color_image.tif')


if __name__ == '__main__':
    app.run(debug=True)
