from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class BaseSVHModel(ABC):
    def __init__(self, input_shape: tuple(32,32,3), num_classes: int = 10, learning_rate: float = 1e-3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Construct and returns the Keras Model Architecture"""
        pass

    def compile(self, model: tf.keras.Model) -> tf.keras.Model:
        model.compile(
                Adam(learning_rate: self.learning_rate)
                loss='categorical_crossentropy'
                metrics=['accuracy']
        )
        return model
