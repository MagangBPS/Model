# %%
import os
import gdown
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Input, Dense, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

# %%
files_to_download = [
    {'id': '1XP7aWc8AqALFA5HofiC6FQOI5rJ6SVc9', 'name': 'Dataset.zip'},
    {'id': '1pzknl3RgnAmMEutPoKJ0iiPcDepDAN8y', 'name': 'idm_baru.csv'}
]

def download_file(file_id, output_file):
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output_file, quiet=False)

for file in files_to_download:
    download_file(file['id'], file['name'])

with zipfile.ZipFile('../Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall("Data/")

# %%
day_path = '../Data/Dataset/Daylight'
night_path = '../Data/Dataset/NTL'
metadata = 'Data/idm_baru.csv'

# %%
idm_df = pd.read_csv(metadata)
idm_df.head()

# %%
idm_df_filtered = idm_df[['KODE BPS', 'KECAMATAN', 'DESA', 'BINARY STATUS']]
idm_df_filtered

# %%
def match_images_with_metadata(image_files, metadata, is_night=False):
    image_data = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        if is_night:
            village_code = img_name.split('.')[0]
        else:
            village_code = img_name.split('-')[0]
        village_info = metadata[metadata['KODE BPS'] == int(village_code)]
        if not village_info.empty:
            kecamatan = village_info['KECAMATAN'].values[0]
            desa = village_info['DESA'].values[0]
            status = village_info['BINARY STATUS'].values[0]
            image_data.append({
                'id' : village_code,
                'filename': img_name,
                'filepath': img_path,
                'kecamatan': kecamatan,
                'desa': desa,
                'status': status
            })
    return pd.DataFrame(image_data)

# %%
day_files = [os.path.join(root, file) 
             for root, _, files in os.walk(day_path) 
             for file in files]
day_files[:10]

# %%
night_files = [os.path.join(root, file) 
               for root, _, files in os.walk(night_path) 
               for file in files]

night_files[:10]

# %%
day_images_metadata = match_images_with_metadata(day_files, idm_df_filtered)
day_images_metadata

# %%
night_images_metadata = match_images_with_metadata(night_files, idm_df_filtered, is_night=True)
night_images_metadata

# %%
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='BINARY STATUS', data=idm_df_filtered)
plt.title('Class Distribution')
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# %%
def show_sample_images(day_image_data, night_image_data, class_label, n=5, cols=2):
    class_images_day = day_image_data[day_image_data['status'] == class_label]
    class_images_night = night_image_data[night_image_data['status'] == class_label]
    
    common_villages = pd.merge(class_images_day, class_images_night, on='id')
    sample_villages = common_villages.sample(n)
    
    rows = (n // cols) + (n % cols > 0)
    fig, axes = plt.subplots(rows, cols*2, figsize=(30, 5*rows))
    fig.patch.set_facecolor('black')  # Set figure background color to black
    axes = axes.flatten()
    
    for i, (_, village_row) in enumerate(sample_villages.iterrows()):
        # Day images
        img_path_day = village_row['filepath_x']
        img_day = Image.open(img_path_day)
        ax_day = axes[i*2]
        ax_day.imshow(img_day)
        ax_day.set_title(f"Day - {class_label} - {village_row['kecamatan_x']}_{village_row['desa_x']}", color='white')
        ax_day.axis('off')
        ax_day.set_facecolor('black')  # Set axes background color to black
        
        # Night images
        img_path_night = village_row['filepath_y']
        img_night = Image.open(img_path_night)
        ax_night = axes[i*2 + 1]
        ax_night.imshow(img_night)
        ax_night.set_title(f"Night - {class_label} - {village_row['kecamatan_y']}_{village_row['desa_y']}", color='white')
        ax_night.axis('off')
        ax_night.set_facecolor('black')  # Set axes background color to black
    
    for ax in axes[len(sample_villages)*2:]:
        ax.axis('off')
        ax.set_facecolor('black')  # Set remaining axes background color to black
    
    plt.tight_layout()
    plt.show()

show_sample_images(day_images_metadata, night_images_metadata, 'MAJU', n=5, cols=2)
show_sample_images(day_images_metadata, night_images_metadata, 'TERTINGGAL', n=5, cols=2)

# %% [markdown]
# ## Balancing Dataset

# %%
def normalize_path(path):
    return os.path.normpath(path)

def normalize_paths(df, path_column):
    df[path_column] = df[path_column].apply(normalize_path)
    return df

# %%
day_images_metadata = normalize_paths(day_images_metadata, 'filepath')
night_images_metadata = normalize_paths(night_images_metadata, 'filepath')

# %%
def augment_images_in_batches(image_paths, metadata_df, datagen, augment_size, batch_size=25, output_dir='Data/Augmented'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    augmented_images = []
    metadata_augmented = []
    num_batches = len(image_paths) // batch_size + (len(image_paths) % batch_size > 0)
    
    for i in range(num_batches):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        batch_metadata = metadata_df[metadata_df['filepath'].isin(batch_paths)]

        for j, img_path in enumerate(batch_paths):
            img = Image.open(img_path).resize((255, 255))
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            x = np.array(img)
            x = x.reshape((1,) + x.shape)
            
            for k, batch in enumerate(datagen.flow(x, batch_size=1)):
                augmented_img = batch[0]
                aug_filename = f"{os.path.basename(img_path).replace('.png', '-a.png')}"
                aug_filepath = os.path.join(output_dir, aug_filename)

                augmented_images.append(augmented_img)
                
                # Save augmented image
                aug_img_pil = Image.fromarray((augmented_img * 255).astype(np.uint8))
                aug_img_pil.save(aug_filepath)

                # Update metadata for augmented images
                original_metadata = batch_metadata.iloc[j]
                metadata_augmented.append({
                    'id': original_metadata['id'],
                    'filename': aug_filename,
                    'filepath': aug_filepath,
                    'kecamatan': original_metadata['kecamatan'],
                    'desa': original_metadata['desa'],
                    'status': original_metadata['status']
                })

                if len(augmented_images) >= augment_size:
                    break

        if len(augmented_images) >= augment_size:
            break

    return augmented_images[:augment_size], pd.DataFrame(metadata_augmented)

# %%
augment_size = len(day_images_metadata[day_images_metadata['status'] == 'TERTINGGAL']) - len(day_images_metadata[day_images_metadata['status'] == 'MAJU'])
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# %%
maju_day_images = day_images_metadata[day_images_metadata['status'] == 'MAJU']['filepath'].tolist()
maju_night_images = night_images_metadata[night_images_metadata['status'] == 'MAJU']['filepath'].tolist()

augmented_day_images, augmented_day_metadata = augment_images_in_batches(maju_day_images, day_images_metadata, datagen, augment_size)
augmented_night_images, augmented_night_metadata = augment_images_in_batches(maju_night_images, night_images_metadata, datagen, augment_size)

# %%
augmented_day_df = pd.DataFrame(augmented_day_metadata)
augmented_night_df = pd.DataFrame(augmented_night_metadata)

# %%
balanced_day_images_metadata = pd.concat([day_images_metadata, augmented_day_df], ignore_index=True)
balanced_night_images_metadata = pd.concat([night_images_metadata, augmented_night_df], ignore_index=True)

# %%
balanced_night_images_metadata

# %%
balanced_day_images_metadata

# %%
def load_images(filenames):
    images = []
    for filename in filenames:
        img = load_img(filename, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

sentinel_filenames = balanced_day_images_metadata[balanced_day_images_metadata['filename'].str.contains('-s')]['filepath'].tolist()
landsat_filenames = balanced_day_images_metadata[balanced_day_images_metadata['filename'].str.contains('-l')]['filepath'].tolist()
night_filenames = balanced_night_images_metadata['filepath'].tolist()

# %%
X_sentinel_images = load_images(sentinel_filenames)
X_landsat_images = load_images(landsat_filenames)
X_night_images = load_images(night_filenames)

# %%
def generator_to_tf_dataset(generator, batch_size):
    output_signature = (
        (tf.TensorSpec(shape=(batch_size, 255, 255, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(batch_size, 255, 255, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature
    )
    
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

train_combined_ds = generator_to_tf_dataset(train_combined_gen, 32)
test_combined_ds = generator_to_tf_dataset(test_combined_gen, 32)

# %%
for (day_batch, night_batch), labels in train_combined_gen:
    print("Day batch shape:", day_batch.shape)
    print("Night batch shape:", night_batch.shape)
    print("Labels shape:", labels.shape)
    break

# %% [markdown]
# ## Train Model

# %%
input_day = Input(shape=(255, 255, 3), name='day_input')
input_night = Input(shape=(255, 255, 3), name='night_input')

def build_cnn_branch(input_tensor):
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    return x

branch_day = build_cnn_branch(input_day)
branch_night = build_cnn_branch(input_night)

combined = Concatenate()([branch_day, branch_night])
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[input_day, input_night], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
history = model.fit(
    train_combined_ds,
    epochs=5,
    validation_data=test_combined_ds
)


