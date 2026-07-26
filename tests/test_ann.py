import numpy as np
import pytest
import tensorflow as tf
from src.models.ann import SVHNANN

def test_ann_output_shape(dummy_batch):
    """Ensures input tensor passes through ANN without shape errors"""
    ann_builder = SVHNANN(input_shape=(32,32,3), num_classes=10)
    model = ann_builder.build()

    predictions = model(dummy_batch, training=False)

    assert predictions.shape == (4,10)

def test_ann_softmax_properties(dummy_batch):
    """Validates that output predictions sum to 1.0 (valid probability)."""
    ann_builder = SVHNANN(input_shape=(32, 32, 3), num_classes=10)
    model = ann_builder.build()

    predictions = model(dummy_batch, training=False).numpy()

    assert not np.isnan(predictions).any()

    prob_sums = np.sum(predictions, axis=-1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

def test_ann_training_step(dummy_batch, dummy_labels):
    """Smoke test: Runs 1 training step to verify backpropagation doesn't crash."""
    ann_builder = SVHNANN(input_shape=(32, 32, 3), num_classes=10)
    model = ann_builder.build()

    history = model.train_on_batch(dummy_batch, dummy_labels)

    loss, accuracy = history[0], history[1]
    assert not np.isnan(loss)
    assert 0.0 <= accuracy <= 1.0
