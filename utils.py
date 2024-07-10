from sklearn.metrics import balanced_accuracy_score
import numpy as np
import os
import glob
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(target_size)
        img = np.array(img).astype(np.float32) / 255.0
    return img

def load_and_preprocess_data(day_folder, night_folder, target_size=(224, 224)):
    x_day = []
    x_night = []
    y = []

    for class_label, class_folder in enumerate(['tertinggal', 'maju']):
        # List all Sentinel and Landsat images for the current class
        sentinel_images = sorted(glob.glob(os.path.join(day_folder, class_folder, '*-s.png')))
        landsat_images = sorted(glob.glob(os.path.join(day_folder, class_folder, '*-l.png')))
        night_images = sorted(glob.glob(os.path.join(night_folder, class_folder, '*.png')))

        # Dictionary to quickly lookup if a day image has a corresponding night image
        night_image_dict = {os.path.basename(img).replace('.png', ''): img for img in night_images}

        for day_image_path in sentinel_images + landsat_images:
            base_name = os.path.basename(day_image_path).replace('-s.png', '').replace('-l.png', '')
            night_image_path = night_image_dict.get(base_name)

            if night_image_path:
                x_day.append(preprocess_image(day_image_path, target_size))
                x_night.append(preprocess_image(night_image_path, target_size))
                y.append(class_label)  # 0 for 'tertinggal', 1 for 'maju'

    x_day = np.array(x_day)
    x_night = np.array(x_night)
    y = np.array(y)

    return x_day, x_night, y

def calculate_balanced_accuracy(model, x_val, y_val):
    y_true = []
    y_pred = []

    # Predict the validation set
    preds = model.predict(x_val)
    y_pred = np.round(preds).flatten()
    y_true = y_val

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return balanced_acc
