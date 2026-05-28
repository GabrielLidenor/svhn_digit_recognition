from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
import config

def build_ann_model():
    """Version 1: Artificial Neural Network Model."""
    model = Sequential([
        Flatten(input_shape=config.INPUT_SHAPE),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model():
    """Version 2: Convolutional Neural Network Model (Best for Images)."""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=config.INPUT_SHAPE),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
