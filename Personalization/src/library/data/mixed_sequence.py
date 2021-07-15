import random
import math

from library.data.sequence import ImageSequence


def personalization_ratios(min_share=2/3, max_share=1.0, min_epochs=30):
    """
    A generator that generates ratios that can be used for adapting the ratios over time. The ratio for
    set A will start with *max_share* and slowly decrease to reach *min_share + 0.0001* after *min_epochs* epochs.

    :param min_share: The minimum share for set A. The default is that set A
    :param max_share: The maximum share for set A.
    :param min_epochs:
    :return:
    """
    index = 0

    while True:
        # Add this offset to x to make sure that f(0) = max_share
        x_offset = -math.log(max_share - min_share)

        # Multiply x by this factor to make sure that f(min_epochs) = min_share + 0.0001 (very close to
        # the target value)
        speed = (math.log(0.0001) - x_offset) / min_epochs

        # Compute the ratio for the two datasets.
        set_a = min_share + math.exp(speed * index - x_offset)
        set_b = 1 - set_a
        index += 1

        yield [set_a, set_b]


class MixedSequence(ImageSequence):
    """A special type of ImageSequence that mixes multiple datasets"""

    def __init__(self, datasets, ratios, epoch_size, batch_size, img_size, charset, max_word_len, augmenter=None):
        """
        :param datasets: A list of datasets to be mixed.
        :param ratios: A list of ratios or a generator.
        :param epoch_size: The number of samples in one epoch.
        :param batch_size: The size of a batch.
        :param img_size: The size of an image.
        :param charset: The characters used with this model.
        :param max_word_len: The maximum length of a word.
        :param augmenter: A list of augmenters to apply to the samples.
        """

        self.datasets = datasets
        self.ratios = ratios
        self.epoch_size = epoch_size

        if isinstance(ratios, list):
            self.ratios = iter(ratios)

        self.epoch = 0
        data = self.prepare_epoch(self.epoch)
        super(MixedSequence, self).__init__(data, batch_size, img_size, charset, max_word_len, augmenter)

    def prepare_epoch(self, epoch):
        """
        Prepare the next epoch by building a new block of data from
        the given datasets using the provided ratios.

        :param epoch: The index of the current epoch.
        :return: Returns the block of data.
        """
        data = list()
        ratios = next(self.ratios)

        for dataset, ratio in zip(self.datasets, ratios):
            num_samples = int(ratio * self.epoch_size)
            data += random.sample(dataset, num_samples)

        random.shuffle(data)
        return data

    def on_epoch_end(self):
        """
        After each epoch put together a new block of data using new
        ratios.
        """
        self.epoch += 1
        self.data = self.prepare_epoch(self.epoch)
