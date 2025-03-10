import os
import numpy as np
import rasterio
from rasterio.plot import show
import math

main_folder = 'LC08_L1TP_144046_20230409_20230420_02_T1'
metadata_file = os.path.join(main_folder, 'LC08_L1TP_144046_20230409_20230420_02_T1_MTL.txt')

output_folder = os.path.join(main_folder, 'Atmospheric_Corrected')
os.makedirs(output_folder, exist_ok=True)

with open(metadata_file, 'r') as file:
    metadata = file.read()

def extract_value(metadata, key):
    start = metadata.find(key) + len(key) + 3
    end = metadata.find('\n', start)
    return metadata[start:end].strip()


sun_elevation = float(extract_value(metadata, 'SUN_ELEVATION'))
earth_sun_distance = float(extract_value(metadata, 'EARTH_SUN_DISTANCE'))

radiance_mult = {}
radiance_add = {}
reflectance_mult = {}
reflectance_add = {}

for band in range(1, 8):
    radiance_mult[f"BAND_{band}"] = float(extract_value(metadata, f'RADIANCE_MULT_BAND_{band}'))
    radiance_add[f"BAND_{band}"] = float(extract_value(metadata, f'RADIANCE_ADD_BAND_{band}'))
    reflectance_mult[f"BAND_{band}"] = float(extract_value(metadata, f'REFLECTANCE_MULT_BAND_{band}'))
    reflectance_add[f"BAND_{band}"] = float(extract_value(metadata, f'REFLECTANCE_ADD_BAND_{band}'))

def process_band(band_number):
    band_file = os.path.join(main_folder, f'LC08_L1TP_144046_20230409_20230420_02_T1_B{band_number}.TIF')

    with rasterio.open(band_file) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile

    radiance = band * radiance_mult[f"BAND_{band_number}"] + radiance_add[f"BAND_{band_number}"]

    #reflectance = radiance * reflectance_mult[f"BAND_{band_number}"] + reflectance_add[f"BAND_{band_number}"]

    theta = math.radians(sun_elevation)
    reflectance = (radiance * math.pi * earth_sun_distance ** 2) / math.sin(theta)

    profile.update(dtype=rasterio.float32)

    output_file = os.path.join(output_folder, f'LC08_L1TP_144046_20230409_20230420_02_T1_B{band_number}_AC.TIF')

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(reflectance, 1)
    print(f'Processed Band {band_number}')


for band_number in range(1, 8):
    process_band(band_number)

print('Atmospheric correction completed.')
