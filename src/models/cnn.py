import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from src.models.base import BaseSVHModel
import config

class SVHNCNN(BaseSVHModel):
    def build(self) -> tf.keras.Model:
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
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
            Dense(self.num_classes, activation='softmax')
        ])

        return self.compile(model)
