import math
import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from library.data.preprocessing import preprocess
from library.data.tokenizer import Tokenizer


class ImageSequence(Sequence):
    """A sequence of batches of training data."""

    def __init__(self, data, batch_size, img_size, charset, max_word_len, augmenter=None, predict=False):
        """
        :param data: The data this sequence will return.
        :param batch_size: The size of a batch returned by this sequence.
        :param img_size: The target size of the images.
        :param charset: The set of characters that the ground truth texts consist of.
        :param max_word_len: The maximum length of a ground truth text.
        :param augmenter: An method of data augmentation to be applied to all images in this sequence.
        :param predict: Whether or not this sequence will be used for prediction or training.
        """
        self.data = data
        self.tokenizer = Tokenizer(charset, max_word_len)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmenter = augmenter
        self.steps = math.ceil(len(self.data) / self.batch_size)
        self.predict = predict

        # The maximum length of the label for each sample in the batch. This will be generated once as this
        # network input is the same for each batch.
        self.x_len = np.asarray([self.tokenizer.maxlen for _ in range(self.batch_size)])

        # The expected output of the network (0 loss). This is generated once as it is the same for each batch.
        self.output = np.zeros(self.batch_size)

    def prepare_train_batch(self, batch):
        """
        Convert the batch into three separate arrays containing input data for the network.
        :param batch: A list of samples.
        :param data_augmentation: Wether or not to use image augmentation.
        :return: Returns three arrays.
        """

        # Create the list containing all images
        x = np.asarray([
            preprocess(
                img=sample.get_img(),
                img_size=self.img_size,
                augmenter=self.augmenter
            ) for sample in batch
        ])

        # Create the list containing the labels encoded as sequences.
        y = np.asarray([self.tokenizer.encode(sample.gt_text) for sample in batch])
        y = pad_sequences(y, maxlen=self.tokenizer.maxlen, padding='post', value=self.tokenizer.blank_label)

        # The maximum length of the label for each sample in the batch.
        x_len = self.x_len[:len(batch)]

        # The actual length of the label (excluding the padding added by pad_sequences)
        y_len = np.asarray([len(sample.gt_text) for sample in batch])

        return x, y, x_len, y_len

    def prepare_predict_batch(self, batch):
        """
        Prepare a batch for for prediction.

        :param batch: The batch to prepare.
        :return: Returns both input and input length for the network.
        """

        # Create the list containing all images
        # Create the list containing all images
        x = np.asarray([
            preprocess(
                img=sample.get_img(),
                img_size=self.img_size,
                augmenter=self.augmenter
            ) for sample in batch
        ])

        # The maximum length of the label for each sample in the batch.
        x_len = self.x_len[:len(batch)]

        return x, x_len

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        """
        :param idx: The current index.
        :return: Returns the next batch with the format depending on the current mode (predict vs train).
        """
        start = idx * self.batch_size
        end = np.minimum((idx + 1) * self.batch_size, len(self.data))
        batch = self.data[start: end]

        if not self.predict:
            x, y, x_len, y_len = self.prepare_train_batch(batch)

            inputs = {
                "input": x,
                "labels": y,
                "input_length": x_len,
                "label_length": y_len
            }

            output = {"CTCloss": self.output[:end - start]}

            return inputs, output
        else:
            x, x_len = self.prepare_predict_batch(batch)

            return x, x_len

    def get_ground_truths(self):
        """
        :return: Returns all labels in a single array.
        """
        return [sample.gt_text for sample in self.data]
