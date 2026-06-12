import os
import joblib
import numpy as np
import pandas as pd
import rasterio
import time

# ==================== 1. Parameter and Path Configuration ====================
model_path_absolute = r''
input_tif_path = r''
csv_path = r''
output_tif_path = r''

print("Initializing and loading the model...")
start_time = time.time()

# Load the LightGBM model
model = joblib.load(model_path_absolute)

# Extract the last 8 digits from the file name (e.g., '20140517')
tif_base_name = os.path.splitext(os.path.basename(input_tif_path))[0]
tif_date_key = tif_base_name[-8:]


# Load CSV table
df = pd.read_csv(csv_path)

# Assume the first column is the identifier column
id_col = df.columns[0]

# Find the row where the last 8 digits of the identifier match tif_date_key
matched_row = df[df[id_col].astype(str).str[-8:] == tif_date_key]

if matched_row.empty:
    raise ValueError(f"Error: No matching record found in CSV for date {tif_date_key}!")

# 1. Extract SUN_ELEVATION (Solar Elevation Angle)
sun_elevation = float(matched_row['SUN_ELEVATION'].values[0])

# 2. Calculate Solar Zenith Angle (SZA) = 90 - Solar Elevation Angle
sza_angle = 90.0 - sun_elevation

# 3. Calculate the cosine of the Solar Zenith Angle: cos(SZA)
sza_cos_value = np.cos(np.radians(sza_angle))

print(f"Match successful -> Sun Elevation: {sun_elevation}°")
print(f"                     Solar Zenith Angle (SZA): {sza_angle}°")
print(f"                     Cosine of SZA: {sza_cos_value:.6f}")

# ==================== 3. Read TIF Image and Process Bands ====================
print("Reading image and constructing feature matrix...")
with rasterio.open(input_tif_path) as src:
    # Get metadata of the original image
    profile = src.profile
    rows, cols = src.height, src.width
    bands_count = src.count

    if bands_count < 6:
        raise ValueError(f"Error: Input image has only {bands_count} bands, 6 bands required!")

    # Read the first 6 bands (shape: 6, height, width)
    img_data = src.read(range(1, 7))

# 1. Flatten into an (n_pixels, 6) matrix
n_pixels = rows * cols
bands_matrix = img_data.transpose(1, 2, 0).reshape((n_pixels, 6))

# 2. Convert to float64 and divide by 10000
bands_matrix = bands_matrix.astype(np.float64) / 10000.0

# 3. Construct an (n_pixels, 1) matrix filled with cos(SZA)
sza_column = np.full((n_pixels, 1), sza_cos_value, dtype=np.float64)

# 4. Concatenate into an (n_pixels, 7) feature matrix [cos(SZA), Band1, Band2, ... Band6]
fp = np.hstack([sza_column, bands_matrix])

# ==================== 4. Model Prediction and Masking ====================
print("Running model prediction...")
predicted_fapar = model.predict(fp)

# Masking: if the value of the 1st band is 0.0, set the predicted value to background (255.0)
zero_mask = (bands_matrix[:, 0] == 0.0)
predicted_fapar[zero_mask] = 255.0

# Reshape back to 2D image (rows, cols)
predicted_image = predicted_fapar.reshape((rows, cols))

# ==================== 5. Save Result as Single-Band GeoTIFF ====================
print("Saving prediction result...")
profile.update(
    dtype=rasterio.float64,
    count=1,
    nodata=255.0
)

with rasterio.open(output_tif_path, 'w', **profile) as dst:
    dst.write(predicted_image, 1)

total_duration = time.time() - start_time
print(f"Processing completed! Result saved to: {output_tif_path}")
print(f"Total time elapsed: {total_duration:.2f} seconds.")