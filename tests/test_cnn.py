import numpy as np
import pytest
import tensorflow as tf
from src.models.cnn import SVHNCNN

def test_cnn_output_shape(dummy_batch):
    """Ensures CNN handles 3D image inputs correctly."""
    cnn_builder = SVHNCNN(input_shape=(32, 32, 3), num_classes=10)
    model = cnn_builder.build()

    predictions = model(dummy_batch, training=False)
    assert predictions.shape == (4, 10)

def test_cnn_overfit_single_sample():
    """Behavioral Test: Checks if CNN can memorize 1 image (verifies gradient flow)."""
    tf.random.set_seed(42)
    single_image = tf.random.uniform((1, 32, 32, 3))
    single_label = tf.one_hot([3], depth=10)

    cnn_builder = SVHNCNN(input_shape=(32, 32, 3), num_classes=10)
    model = cnn_builder.build()

    for _ in range(15):
        model.train_on_batch(single_image, single_label)

    final_pred = model(single_image, training=False).numpy()
    predicted_class = np.argmax(final_pred, axis=-1)[0]

    assert predicted_class == 3
