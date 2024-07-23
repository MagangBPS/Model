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
def load_and_preprocess_data(day_folder, target_size=(224, 224)):
    day_images = sorted(glob.glob(os.path.join(day_folder, '*-s.png')))

    x_day, y = [], []
    desa_ids = []

    for day_image_path in day_images:
        base_name = os.path.basename(day_image_path).replace('-s.png', '')
        desa_id = base_name.split('-')[0]

        # Exclude 2022 images
        if desa_id == '2022':
            continue

        # Pilih salah satu
        day_image_path = os.path.join(day_folder, f'{base_name}-l.png') # Change to -s or -l for Sentinel or Landsat images

        if os.path.exists(day_image_path):
            day_image = preprocess_image(day_image_path, target_size)

            x_day.append(day_image)
            desa_ids.append(desa_id)

            if 'MAJU' in day_image_path:
                y.append(1)
            else:
                y.append(0)

    x_day = np.array(x_day)
    y = np.array(y)
    desa_ids = np.array(desa_ids)

    return x_day, y, desa_ids

# Load and preprocess data
day_folder_maju = '../Dataset/Day/MAJU'
day_folder_tertinggal = '../Dataset/Day/TERTINGGAL'

x_day_maju, y_maju, desa_ids_maju = load_and_preprocess_data(day_folder_maju)
x_day_tertinggal, y_tertinggal, desa_ids_tertinggal = load_and_preprocess_data(day_folder_tertinggal)

# Combine the datasets
x_day = np.concatenate([x_day_maju, x_day_tertinggal], axis=0)
y = np.concatenate([y_maju, y_tertinggal], axis=0)
desa_ids = np.concatenate([desa_ids_maju, desa_ids_tertinggal], axis=0)

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
    'status': np.where(predictions == 1, 'Maju', 'Tertinggal'),
    'asli': y,
    'status_asli': np.where(y == 1, 'Maju', 'Tertinggal')
})

# Save the results to a CSV file
results_df.to_csv('../dataframe/hasil_prediksi.csv', index=False)

print("Predictions saved to 'dataframe/hasil_prediksi.csv'")
