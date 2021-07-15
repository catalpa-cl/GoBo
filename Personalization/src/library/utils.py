import os
import random
import numpy as np
from imgaug import seed
from library.const import *

from library.model.model import HTRModel
from library.model.architecture import attention
from library.data.sequence import ImageSequence


def load_model(checkpoint=None, use_conv=True, use_lstm=True):
    """
    Load the model and a checkpoint if specified.

    :param checkpoint: The checkpoint to load for this model.
    :param use_conv: Whether to set the convolutional layers to be trainable
    :param use_lstm: Whether to set the LSTM layers to be trainable
    :return: Returns the loaded model.
    """
    # Set up the architecture.
    in_data, out_data, optimizer = attention(Const.input_size,
                                             Const.output_size,
                                             lr=Const.learning_rate,
                                             use_conv=use_conv,
                                             use_lstm=use_lstm)

    # Set up the handwriting recognition model.
    model = HTRModel(inputs=in_data, outputs=out_data)

    # Compile the model.
    model.compile(optimizer=optimizer)

    # Load weights if requested.
    if checkpoint:
        model.load_checkpoint(checkpoint)

    return model


def to_sequences(train_samples, valid_samples, augmenter=None):
    """
    :param train_samples: The training data to create a sequence for.
    :param valid_samples: The validation data to create a sequence for.
    :param augmenter: The augmenter for the training sequence.
    :return: Returns both the resulting training and validation sequence.
    """
    train_seq = to_train_sequence(train_samples, augmenter)
    valid_seq = to_valid_sequence(valid_samples)

    return train_seq, valid_seq


def to_train_sequence(samples, augmenter=None):
    """
    :param samples: The training data to create a sequence for.
    :param augmenter: The augmenter for the training sequence.
    :return: Returns the resulting training sequence.
    """
    train_seq = ImageSequence(
        data=samples,
        batch_size=Const.batch_size,
        img_size=Const.img_size,
        charset=Const.charset,
        max_word_len=Const.max_text_length,
        augmenter=augmenter,
    )

    return train_seq


def to_valid_sequence(samples):
    """
    :param samples: The validation data to create a sequence for.
    :return: Returns the resulting validation sequence.
    """
    valid_seq = ImageSequence(
        data=samples,
        batch_size=Const.batch_size,
        img_size=Const.img_size,
        charset=Const.charset,
        max_word_len=Const.max_text_length,
        augmenter=None,
    )

    return valid_seq


def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(55)
    np.random.seed(55)
    random.seed(55)
    # tf.random.set_seed(55)
    seed(entropy=55)

