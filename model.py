from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, concatenate

def create_model(input_shape_day=(224, 224, 3), input_shape_night=(224, 224, 3)):
    # Day input
    input_day = Input(shape=input_shape_day, name='day_input')
    x1 = Conv2D(32, (3, 3), activation='relu')(input_day)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Flatten()(x1)

    # Night input
    input_night = Input(shape=input_shape_night, name='night_input')
    x2 = Conv2D(32, (3, 3), activation='relu')(input_night)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Flatten()(x2)

    # Combined model
    combined = concatenate([x1, x2])
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_day, input_night], outputs=output)
    return model
