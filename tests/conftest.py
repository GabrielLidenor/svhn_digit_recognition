import pytest
import tensorflow as tf

@pytest.fixture
def dummy_batch():
    """Generates a mock batch of 4 RGB images (32x32x3)."""
    return tf.random.uniform(shape=(4, 32, 32, 3), minval=0.0, maxval=1.0)

@pytest.fixture
def dummy_labels():
    """Generates mock one-hot encoded labels for 10 classes."""
    indices = [0, 1, 2, 3]
    return tf.one_hot(indices, depth=10)
