import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model
from BalancedAccuracy import BalancedAccuracy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score
import seaborn as sns

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#PERBAIKI!!
def plot_random_pairs(x_day, x_night, y, num_pairs=2):
    import random
    # Get random indices
    indices = random.sample(range(len(x_day)), num_pairs)

    plt.figure(figsize=(8, 4 * num_pairs))

    for i, idx in enumerate(indices):
        label = 'MAJU' if y[idx] == 1 else 'TERTINGGAL'

        plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(x_day[idx])
        plt.title(f'Day Image\nLabel: {label}', fontsize=10)
        plt.axis('off')

        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(x_night[idx], cmap='gray')
        plt.title(f'Night Image\nLabel: {label}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


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
def load_and_preprocess_data(day_folder, night_folder, target_size=(224, 224)):
    day_images = sorted(glob.glob(os.path.join(day_folder, '*-s.png')))
    night_images = sorted(glob.glob(os.path.join(night_folder, '*.png')))

    x_day_s, x_day_l, x_night, y = [], [], [], []

    for day_image_path in day_images:
        base_name = os.path.basename(day_image_path).replace('-s.png', '')
        sentinel_day_image_path = os.path.join(day_folder, f'{base_name}-s.png')
        landsat_day_image_path = os.path.join(day_folder, f'{base_name}-l.png')
        night_image_path = os.path.join(night_folder, f'{base_name}.png')

        if os.path.exists(sentinel_day_image_path) and os.path.exists(night_image_path):
            day_image_s = preprocess_image(sentinel_day_image_path, target_size)
            day_image_l = preprocess_image(landsat_day_image_path, target_size)
            night_image = preprocess_image(night_image_path, target_size)

            x_day_s.append(day_image_s)
            x_day_l.append(day_image_l)
            x_night.append(night_image)

            if 'MAJU' in sentinel_day_image_path:
                y.append(1)
            else:
                y.append(0)

    x_day_s = np.array(x_day_s)
    x_day_l = np.array(x_day_l)
    x_night = np.array(x_night)
    y = np.array(y)

    return x_day_s, x_day_l, x_night, y


def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you should add them pass them as a dictionary.
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''

    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    return new_model

#Create Model Architecture
original_model1 = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)
densenet_day = add_prefix(original_model1, 'densenet_day_')

original_model2 = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)
densenet_night = add_prefix(original_model2, 'densenet_night_')

def create_model(input_shape=(224, 224, 3)):
    # Define input layers
    input_day = Input(shape=input_shape, name='input_day')
    input_night = Input(shape=input_shape, name='input_night')

    # Use DenseNet121 as a feature extractor for both day and night images
    feature_extractor_day = densenet_day(input_day)
    feature_extractor_night = densenet_night(input_night)

    # Global pooling layers
    day_output = GlobalAveragePooling2D(name='day_global_avg_pool')(feature_extractor_day)
    night_output = GlobalAveragePooling2D(name='night_global_avg_pool')(feature_extractor_night)

    # Concatenate the outputs
    combined_output = Concatenate(name='concatenate')([day_output, night_output])

    # Fully connected layers
    dense = Dense(256, activation='relu', name='dense_256')(combined_output)
    dense = Dense(128, activation='relu', name='dense_128')(dense)
    # dropout = Dropout(0.5, name='dropout')(dense)

    # Output layer for binary classification
    output = Dense(1, activation='sigmoid', name='output')(dense)

    # Create the model
    model = Model(inputs=[input_day, input_night], outputs=output, name='day_night_model')

    return model

# Load and preprocess data
day_folder_maju = '../Dataset/Day/MAJU'
night_folder_maju = '../Dataset/Night/MAJU'

x_day_s_maju, x_day_l_maju, x_night_maju, y_maju = load_and_preprocess_data(day_folder_maju, night_folder_maju)

day_folder_tertinggal = '../Dataset/Day/TERTINGGAL'
night_folder_tertinggal = '../Dataset/Night/TERTINGGAL'

x_day_s_tertinggal, x_day_l_tertinggal, x_night_tertinggal, y_tertinggal = load_and_preprocess_data(day_folder_tertinggal, night_folder_tertinggal)

# Concatenate data
x_day = np.concatenate([x_day_s_maju, x_day_s_tertinggal, x_day_l_maju, x_day_l_tertinggal], axis=0)
x_night = np.concatenate([x_night_maju, x_night_tertinggal, x_night_maju, x_night_tertinggal], axis=0)
y = np.concatenate([y_maju, y_tertinggal, y_maju, y_tertinggal], axis=0)

# # Plot 2 random pairs
# plot_random_pairs(x_day, x_night, y, num_pairs=2)

# Split the dataset into training and validation sets
x_day_train, x_day_val, x_night_train, x_night_val, y_train, y_val = train_test_split(
    x_day, x_night, y, test_size=0.2, random_state=42)

# Check Data Shape
print (f"""
        x_day_train: {x_day_train.shape}
        x_day_val: {x_day_val.shape}
        x_night_train: {x_night_train.shape}
        x_night_val: {x_night_val.shape}
        y_train: {y_train.shape}
        y_val: {y_val.shape}
        """)

# Create model
model = create_model(input_shape=(224, 224, 3))

#Visualize the model architecture
plot_model(model, to_file='model_architecture_nodropout.png', show_shapes=True, show_layer_names=True)

# Hyperparams
EPOCHS = 30
LR = 0.00005

#Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='binary_crossentropy',
              metrics=['accuracy', BalancedAccuracy()])

# #Define callbacks
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
#     ModelCheckpoint('../h5_models/status_desa_densenet121_best.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# ]

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Fit the model
history = model.fit(
    [x_day_train, x_night_train],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    validation_data=([x_day_val, x_night_val], y_val),
    class_weight=class_weights,
    # callbacks=callbacks,
    verbose=1
)

# Plot training history
def plot_training_history(history):
    # Loss
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history.history['balanced_accuracy'], label='Train Balanced Accuracy', color='blue')
    plt.plot(history.history['val_balanced_accuracy'], label='Val Balanced Accuracy', color='orange')
    plt.title('Training and Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Get Confusion Matrix
def evaluate_model(model, x_day, x_night, y_true, dataset_type="Validation"):
    # Predict the output
    y_pred_prob = model.predict([x_day, x_night])
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{dataset_type} Data Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{dataset_type} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Evaluate on training data
evaluate_model(model, x_day_train, x_night_train, y_train, dataset_type="Training")

# Evaluate on validation data
evaluate_model(model, x_day_val, x_night_val, y_val, dataset_type="Validation")

# Save the final model
model.save(f'../h5_models/status_desa_densenet121_{EPOCHS}epochs_nodropout.h5')
print("Model saved successfully.")