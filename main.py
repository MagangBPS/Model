import os
from download_and_extract import download_and_extract_files
from data_preprocessing import preprocess_metadata, match_images_with_metadata, normalize_paths
from visualization import plot_class_distribution, show_sample_images
from background_modification import process_images
from utils import calculate_balanced_accuracy, load_and_preprocess_data
from model import create_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_config import setup_tensorflow, clear_session

setup_tensorflow()

# # Download and extract files
# files_to_download = [
#     {'id': '1XP7aWc8AqALFA5HofiC6FQOI5rJ6SVc9', 'name': 'Dataset.zip'},
#     {'id': '1pzknl3RgnAmMEutPoKJ0iiPcDepDAN8y', 'name': 'idm_baru.csv'}
# ]
# download_and_extract_files(files_to_download)

# Paths
day_path = 'Data/Dataset/Daylight'
night_path = 'Data/Dataset/NTL'
metadata_path = 'Data/idm_baru.csv'
modified_day_path = 'Data/Modified_Dataset/Daylight'
modified_night_path = 'Data/Modified_Dataset/NTL'

# Preprocess metadata
idm_df_filtered = preprocess_metadata(metadata_path)

# Match images with metadata
day_files = [os.path.join(root, file) for root, _, files in os.walk(day_path) for file in files]
night_files = [os.path.join(root, file) for root, _, files in os.walk(night_path) for file in files]
day_images_metadata = match_images_with_metadata(day_files, idm_df_filtered)
night_images_metadata = match_images_with_metadata(night_files, idm_df_filtered, is_night=True)
day_images_metadata = normalize_paths(day_images_metadata, 'filepath') # path \
night_images_metadata = normalize_paths(night_images_metadata, 'filepath')

# # Visualize data
# plot_class_distribution(idm_df_filtered)
# show_sample_images(day_images_metadata, night_images_metadata, 'MAJU', n=5, cols=2)
# show_sample_images(day_images_metadata, night_images_metadata, 'TERTINGGAL', n=5, cols=2)

# # Modify background color
# process_images(day_path, modified_day_path)
# process_images(night_path, modified_night_path)

# Load and preprocess data
x_day, x_night, y = load_and_preprocess_data(modified_day_path, modified_night_path)

# Split the dataset into training and validation sets
x_day_train, x_day_val, x_night_train, x_night_val, y_train, y_val = train_test_split(
    x_day, x_night, y, test_size=0.3, random_state=42)

# Create model
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('status_desa_cnn_best.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

# Fit the model
history = model.fit(
    [x_day_train, x_night_train],
    y_train,
    epochs=1,
    batch_size=16,
    validation_data=([x_day_val, x_night_val], y_val),
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('status_desa_cnn.h5')
print("Model saved successfully.")

# Calculate and print balanced accuracy
balanced_acc = calculate_balanced_accuracy(model, [x_day_val, x_night_val], y_val)
print(f'Balanced Accuracy: {balanced_acc * 100:.2f}%')

clear_session()