import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from src.models.base import BaseSVHModel
import config

class SVHNANN(BaseSVHModel):
    def build(self) -> tf.keras.Model:
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        return self.compile(model)



