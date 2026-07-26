import tensorflow as tf
import tensorflow.keras import layers, models
import src.models.base import BaseSVHModel

class SVHNANN(BaseSVHModel):
    def build(self) -> tf.keras.Model:
        model = Sequential([
            Flatten(input_shape=config.INPUT_SHAPE),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(config.NUM_CLASSES, activation='softmax')
        ])

        return self.compile(model)



