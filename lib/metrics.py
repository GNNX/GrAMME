import tensorflow as tf
import numpy as np


def regularization_loss(Weights, lambd=0.005):
    assert len(Weights) > 0
    reg = tf.add_n([tf.nn.l2_loss(weight) for weight in Weights]) * lambd
    return reg


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Compute accuracy outside of TF."""
    preds = preds[mask]
    labels = labels[mask]
    num_correct_pred = np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))

    return num_correct_pred / np.sum(mask)
