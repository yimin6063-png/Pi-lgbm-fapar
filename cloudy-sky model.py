import os
import pickle
import numpy as np
import scipy.io as sio
import rasterio
import time

# ==================== 1. Parameter and Path Configuration ====================
model_path = r''
input_mat_path = r''
reference_tif_path = r''
output_tif_path = r''

print("Initializing and loading the model...")
start_time = time.time()

# Load the LightGBM model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ==================== 2. Read Reference TIF Spatial Dimensions ====================
print("Reading spatial dimensions from reference SR TIF...")
with rasterio.open(reference_tif_path) as src:
    # Get metadata profile and 2D spatial dimensions (height and width)
    profile = src.profile
    rows, cols = src.height, src.width
    print(f"Reference image dimensions: {rows} rows x {cols} columns")

# ==================== 3. Load and Process MAT Data ====================
print("Loading and processing MAT feature data...")
mat_data = sio.loadmat(input_mat_path)
X = mat_data.get('resultData')

if X is None:
    raise ValueError(f"Error: Variable 'resultData' not found in {input_mat_path}!")

X = X[:, [0, 1, 2, 3, 4, 6, 7, 8]]

if X.size == 0:
    raise ValueError(f"Error: Data matrix 'resultData' is empty!")
y_pred = np.zeros(X.shape[0], dtype=np.float32)

mask_255 = np.all(X[:, :5] == 255, axis=1)

y_pred[mask_255] = 255.0

valid_indices = ~mask_255
if np.any(valid_indices):
    print("Running model prediction for valid pixels...")
    y_pred[valid_indices] = model.predict(X[valid_indices], num_iteration=model.best_iteration)

predicted_image = y_pred.reshape((rows, cols), order='F')

# ==================== 5. Save Result as Single-Band GeoTIFF ====================
print("Saving prediction result as a single-band GeoTIFF...")
# Force the output profile to be single-band (count=1) while preserving spatial reference
profile.update(
    dtype=rasterio.float32,
    count=1,
    nodata=255.0
)

with rasterio.open(output_tif_path, 'w', **profile) as dst:
    # Write the 2D array into the first (and only) band
    dst.write(predicted_image, 1)

total_duration = time.time() - start_time
print(f"Processing completed! Result saved to: {output_tif_path}")
print(f"Total time elapsed: {total_duration:.2f} seconds.")