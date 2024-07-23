import numpy as np
import pandas as pd
import os
import cv2
import glob
from tensorflow.keras.models import load_model
from tqdm import tqdm
from BalancedAccuracy import BalancedAccuracy

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        rgb_channels = image[:, :, :3]
        black_background = np.zeros_like(rgb_channels, dtype=np.uint8)
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        image = cv2.convertScaleAbs(rgb_channels * alpha_factor + black_background * (1 - alpha_factor))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0

    return image

# Function to load and preprocess data
def load_and_preprocess_data(folder, data_type, target_size=(224, 224)):
    images_folder = os.path.join(folder, data_type)
    image_paths = sorted(glob.glob(os.path.join(images_folder, '*.png')))

    x_day = []
    desa_ids = []

    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        desa_id = base_name.split('.')[0]

        if os.path.exists(image_path):
            image = preprocess_image(image_path, target_size)
            x_day.append(image)
            desa_ids.append(desa_id)

    x_day = np.array(x_day)
    desa_ids = np.array(desa_ids)

    return x_day, desa_ids

# Load and preprocess data from Landsat8 or Sentinel2 folder
data_folder = '../Data2024'
data_type = 'Sentinel2'  # Choose 'Landsat8' or 'Sentinel2'
x_day, desa_ids = load_and_preprocess_data(data_folder, data_type)

# Load the trained model
custom_objects = {'BalancedAccuracy': BalancedAccuracy}
model = load_model('../h5_models/status_desa_densenet121_day_only_30epochs_dropout.h5', custom_objects=custom_objects)

# Perform predictions in batches
batch_size = 32  # Adjust the batch size as needed
predictions = []

for start in tqdm(range(0, len(x_day), batch_size)):
    end = start + batch_size
    batch_x_day = x_day[start:end]
    batch_predictions = model.predict(batch_x_day)
    predictions.extend(batch_predictions)

# Convert predictions to binary classification
predictions = (np.array(predictions) > 0.5).astype("int32").flatten()

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'id_desa': desa_ids,
    'prediksi': predictions,
    'status_prediksi': np.where(predictions == 1, 'Maju', 'Tertinggal')
})

# Save the results to a CSV file
output_csv = f'../dataframe/hasil_prediksi_{data_type}_2024.csv'
results_df.to_csv(output_csv, index=False)

print(f"Predictions saved to '{output_csv}'")
