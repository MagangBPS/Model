from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_data_generators(day_path, night_path, batch_size=32, target_size=(256, 256)):
    datagen_day = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    datagen_night = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator_day_landsat = datagen_day.flow_from_directory(
        directory=day_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        classes=['landsat']
    )

    train_generator_day_sentinel = datagen_day.flow_from_directory(
        directory=day_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        classes=['sentinel']
    )

    validation_generator_day_landsat = datagen_day.flow_from_directory(
        directory=day_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        classes=['landsat']
    )

    validation_generator_day_sentinel = datagen_day.flow_from_directory(
        directory=day_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        classes=['sentinel']
    )

    train_generator_night = datagen_night.flow_from_directory(
        directory=night_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator_night = datagen_night.flow_from_directory(
        directory=night_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Combine day generators and night generator
    train_generator_day = zip(train_generator_day_landsat, train_generator_day_sentinel)
    validation_generator_day = zip(validation_generator_day_landsat, validation_generator_day_sentinel)

    return train_generator_day, validation_generator_day, train_generator_night, validation_generator_night

class CombinedDataGenerator:
    def __init__(self, day_gen, night_gen, batch_size):
        self.day_gen = day_gen
        self.night_gen = night_gen
        self.batch_size = batch_size

    def __len__(self):
        return min(len(self.day_gen), len(self.night_gen))

    def __iter__(self):
        while True:
            day_batch = next(self.day_gen)
            night_batch = next(self.night_gen)

            # Combine day and night images
            day_images = np.concatenate([day_batch[0][0], day_batch[1][0]], axis=-1)  # Combine Landsat and Sentinel
            night_images = night_batch[0]
            day_labels = day_batch[1]
            night_labels = night_batch[1]

            # Create combined images and labels
            combined_images = [day_images, night_images]
            yield combined_images, day_labels
