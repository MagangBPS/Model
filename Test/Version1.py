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
                'filename': img_name,
                'filepath': img_path,
                'kecamatan': kecamatan,
                'desa': desa,
                'status': status
            })
    return pd.DataFrame(image_data)

# %%
day_files = []
for root, dirs, files in os.walk(day_path):
    for file in files:
        day_files.append(os.path.join(root, file))

day_files[:10]

# %%
night_files = []
for root, dirs, files in os.walk(night_path):
    for file in files:
        night_files.append(os.path.join(root, file))

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
    
    sample_images_day = class_images_day.sample(n)
    sample_images_night = class_images_night.sample(n)
    
    rows = (n // cols) + (n % cols > 0)
    fig, axes = plt.subplots(rows, cols*2, figsize=(30, 5*rows))
    axes = axes.flatten()
    
    for i, ((_, day_row), (_, night_row)) in enumerate(zip(sample_images_day.iterrows(), sample_images_night.iterrows())):
        # Day images
        img_path_day = day_row['filepath']
        img_day = Image.open(img_path_day)
        ax_day = axes[i*2]
        ax_day.imshow(img_day)
        ax_day.set_title(f"Day - {class_label} - {day_row['kecamatan']}_{day_row['desa']}")
        ax_day.axis('off')
        
        # Night images
        img_path_night = night_row['filepath']
        img_night = Image.open(img_path_night)
        ax_night = axes[i*2 + 1]
        ax_night.imshow(img_night)
        ax_night.set_title(f"Night - {class_label} - {night_row['kecamatan']}_{night_row['desa']}")
        ax_night.axis('off')
    
    for ax in axes[len(sample_images_day)*2:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

show_sample_images(day_images_metadata, night_images_metadata, 'MAJU', n=5, cols=2)
show_sample_images(day_images_metadata, night_images_metadata, 'TERTINGGAL', n=5, cols=2)


# %% [markdown]
# ## Balancing Dataset

# %% [markdown]
# ### Tes 1

# %%
def augment_images_in_batches(image_paths, datagen, augment_size, batch_size=25):
    augmented_images = []
    num_batches = len(image_paths) // batch_size + (len(image_paths) % batch_size > 0)
    
    for i in range(num_batches):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            x = np.array(img)
            x = x.reshape((1,) + x.shape)
            
            for _ in range(augment_size // batch_size):
                for batch in datagen.flow(x, batch_size=1):
                    batch_images.append(batch[0])
                    break
        
        augmented_images.extend(batch_images)
        
        if len(augmented_images) >= augment_size:
            break
    
    return augmented_images[:augment_size]

# %%
augment_size = len(day_images_metadata[day_images_metadata['status'] == 'TERTINGGAL']) - len(day_images_metadata[day_images_metadata['status'] == 'MAJU'])

maju_day_images = day_images_metadata[day_images_metadata['status'] == 'MAJU']['filepath'].tolist()
maju_night_images = night_images_metadata[night_images_metadata['status'] == 'MAJU']['filepath'].tolist()

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
augmented_day_images = augment_images_in_batches(maju_day_images, datagen, augment_size)
augmented_night_images = augment_images_in_batches(maju_night_images, datagen, augment_size)

# %%
augmented_day_df = pd.DataFrame({
    'filename': [f"aug_day_{i}.png" for i in range(len(augmented_day_images))],
    'kecamatan': ['Augmented'] * len(augmented_day_images),
    'desa': ['Augmented'] * len(augmented_day_images),
    'status': ['MAJU'] * len(augmented_day_images)
})

augmented_night_df = pd.DataFrame({
    'filename': [f"aug_night_{i}.png" for i in range(len(augmented_night_images))],
    'kecamatan': ['Augmented'] * len(augmented_night_images),
    'desa': ['Augmented'] * len(augmented_night_images),
    'status': ['MAJU'] * len(augmented_night_images)
})

# %%
balanced_day_images_metadata = pd.concat([day_images_metadata, augmented_day_df])
balanced_night_images_metadata = pd.concat([night_images_metadata, augmented_night_df])

# %%
X_day_train, X_day_test, y_train, y_test = train_test_split(balanced_day_images_metadata['filename'], balanced_day_images_metadata['status'], test_size=0.2, stratify=balanced_day_images_metadata['status'], random_state=42)
X_night_train, X_night_test, _, _ = train_test_split(balanced_night_images_metadata['filename'], balanced_night_images_metadata['status'], test_size=0.2, stratify=balanced_night_images_metadata['status'], random_state=42)

# %%
def create_data_generator(filenames, labels, batch_size, datagen):
    df = pd.DataFrame({'filename': filenames, 'label': labels}).reset_index(drop=True)
    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        target_size=(255, 255),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )
    return generator

train_gen_day = create_data_generator(X_day_train, y_train, 32, datagen)
test_gen_day = create_data_generator(X_day_test, y_test, 32, datagen)

train_gen_night = create_data_generator(X_night_train, y_train, 32, datagen)
test_gen_night = create_data_generator(X_night_test, y_test, 32, datagen)

# %% [markdown]
# ### Tes 2

# %%
# Data Augmentation
augment_size = len(day_images_metadata[day_images_metadata['status'] == 'TERTINGGAL']) - len(day_images_metadata[day_images_metadata['status'] == 'MAJU'])

maju_day_images = day_images_metadata[day_images_metadata['status'] == 'MAJU']['filepath'].tolist()
maju_night_images = night_images_metadata[night_images_metadata['status'] == 'MAJU']['filepath'].tolist()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_day_images = augment_images_in_batches(maju_day_images, datagen, augment_size)
augmented_night_images = augment_images_in_batches(maju_night_images, datagen, augment_size)

augmented_day_df = pd.DataFrame({
    'filename': [f"aug_day_{i}.png" for i in range(len(augmented_day_images))],
    'kecamatan': ['Augmented'] * len(augmented_day_images),
    'desa': ['Augmented'] * len(augmented_day_images),
    'status': ['MAJU'] * len(augmented_day_images)
})

augmented_night_df = pd.DataFrame({
    'filename': [f"aug_night_{i}.png" for i in range(len(augmented_night_images))],
    'kecamatan': ['Augmented'] * len(augmented_night_images),
    'desa': ['Augmented'] * len(augmented_night_images),
    'status': ['MAJU'] * len(augmented_night_images)
})

balanced_day_images_metadata = pd.concat([day_images_metadata, augmented_day_df])
balanced_night_images_metadata = pd.concat([night_images_metadata, augmented_night_df])

# %%
X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(balanced_day_images_metadata['filename'], balanced_day_images_metadata['status'], test_size=0.2, stratify=balanced_day_images_metadata['status'], random_state=42)
X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(balanced_night_images_metadata['filename'], balanced_night_images_metadata['status'], test_size=0.2, stratify=balanced_night_images_metadata['status'], random_state=42)

# %%
def create_data_generator(filenames, labels, batch_size, datagen):
    df = pd.DataFrame({'filename': filenames, 'label': labels}).reset_index(drop=True)
    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        target_size=(255, 255),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )
    return generator

train_gen_day = create_data_generator(X_day_train, y_day_train, 32, datagen)
test_gen_day = create_data_generator(X_day_test, y_day_test, 32, datagen)

train_gen_night = create_data_generator(X_night_train, y_night_train, 32, datagen)
test_gen_night = create_data_generator(X_night_test, y_night_test, 32, datagen)

# %%
def combined_generator(day_gen, night_gen):
    while True:
        day_images, day_labels = next(day_gen)
        night_images, night_labels = next(night_gen)
        
        yield [day_images, night_images], day_labels  # assuming day_labels and night_labels are the same

# Create the combined generators
train_combined_gen = combined_generator(train_gen_day, train_gen_night)
val_combined_gen = combined_generator(test_gen_day, test_gen_night)

# %% [markdown]
# ### Tes 3

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
def augment_images_in_batches(image_paths, datagen, augment_size, batch_size=25):
    augmented_images = []
    num_batches = len(image_paths) // batch_size + (len(image_paths) % batch_size > 0)
    
    for i in range(num_batches):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).resize((255, 255))
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            x = np.array(img)
            x = x.reshape((1,) + x.shape)
            
            for _ in range(augment_size // batch_size):
                for batch in datagen.flow(x, batch_size=1):
                    batch_images.append(batch[0])
                    break
        
        augmented_images.extend(batch_images)
        
        if len(augmented_images) >= augment_size:
            break
    
    return augmented_images[:augment_size]

# %%
maju_day_images = day_images_metadata[day_images_metadata['status'] == 'MAJU']['filepath'].tolist()
maju_night_images = night_images_metadata[night_images_metadata['status'] == 'MAJU']['filepath'].tolist()

augmented_day_images = augment_images_in_batches(maju_day_images, datagen, augment_size)
augmented_night_images = augment_images_in_batches(maju_night_images, datagen, augment_size)

# %%
augmented_day_df = pd.DataFrame({
    'filename': [f"aug_day_{i}.png" for i in range(len(augmented_day_images))],
    'kecamatan': ['Augmented'] * len(augmented_day_images),
    'desa': ['Augmented'] * len(augmented_day_images),
    'status': ['MAJU'] * len(augmented_day_images)
})

augmented_night_df = pd.DataFrame({
    'filename': [f"aug_night_{i}.png" for i in range(len(augmented_night_images))],
    'kecamatan': ['Augmented'] * len(augmented_night_images),
    'desa': ['Augmented'] * len(augmented_night_images),
    'status': ['MAJU'] * len(augmented_night_images)
})

# %%
balanced_day_images_metadata = pd.concat([day_images_metadata, augmented_day_df])
balanced_night_images_metadata = pd.concat([night_images_metadata, augmented_night_df])

# %%
X_day = balanced_day_images_metadata['filename']
y_day = balanced_day_images_metadata['status']

X_night = balanced_night_images_metadata['filename']
y_night = balanced_night_images_metadata['status']

X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(X_day, y_day, test_size=0.2, random_state=42, stratify=y_day)
X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(X_night, y_night, test_size=0.2, random_state=42, stratify=y_night)

# %%
print(f'Day training set size: {len(X_day_train)}')
print(f'Day testing set size: {len(X_day_test)}')
print(f'Night training set size: {len(X_night_train)}')
print(f'Night testing set size: {len(X_night_test)}')

# %%
def load_images(filenames, base_path):
    images = []
    for filename in filenames:
        img_path = os.path.join(base_path, filename)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

X_day_train_images = load_images(X_day_train, day_path)
X_day_test_images = load_images(X_day_test, day_path)
X_night_train_images = load_images(X_night_train, night_path)
X_night_test_images = load_images(X_night_test, night_path)

# %%
def combined_data_generator(day_gen, night_gen):
    while True:
        day_batch = next(day_gen)
        night_batch = next(night_gen)
        
        yield (day_batch[0], night_batch[0]), day_batch[1]

train_combined_gen = combined_data_generator(train_gen_day, train_gen_night)
test_combined_gen = combined_data_generator(test_gen_day, test_gen_night)

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


