import os
import cv2
import numpy as np

from library.data.loading import display_loading_bar
from library.data.data_augmentation.transformation import affine_elastic


# initialize the pipeline once.
pipeline = affine_elastic()


def load_lexicon(file_path):
    """
    Loads all the words from the given file.

    :param file_path: The path of the file containing the words.
    :return: Returns a list of words.
    """

    with open(file_path) as f:
        words = f.read().split(' ')

    return words


def load_char_list(path):
    """
    Load a char list for decoding and encoding sequences.

    :param path: The path of the char list.
    :return: Returns a list of chars.
    """

    with open(path) as f:
        line = f.read()

    return line


def load_img(path):
    """
    Load and resize an image.

    :param path: The path of the image to be loaded.
    :param img_size: The target shape for the image.
    :return: Returns an array containing the image data.
    """

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_labels(samples):
    """
    :param samples: A list of samples.
    :return: Returns a list containing all labels of the given list of samples.
    """
    return [s.gt_text for s in samples]


class Sample:
    """
    Contains both the label and the path for the image
    """

    def __init__(self, gt_text, img_path=None, img=None):
        """
        :param gt_text: The label.
        :param img_path: The path of the image.
        :param img: The image itself.
        """

        self.gt_text = gt_text
        self.img_path = img_path
        self.img = img

    def get_img(self):
        """
        Loads the image if it hasn't been loaded yet.

        :param img_size: The target shape of the image.
        :param data_augmentation: wether or not to apply data augmentation to the image.
        :return: Returns the image as a numpy array.
        """

        return load_img(self.img_path)

    def __hash__(self):
        return self.img_path.__hash__()

    def __str__(self):
        return self.gt_text + ' at ' + self.img_path

    def __eq__(self, other):
        return self.img_path == other.img_path


def samples_to_dataset(path, samples):
    """
    Converts a list of samples to a dataset that can be loaded.
    :param path: The output path for the dataset.
    :param samples: The samples to add to the dataset.
    """

    with open(path, 'w') as f:
        for sample in samples:
            f.write(sample.img_path + ' ok ' + sample.gt_text + '\n')


def adaptContrast(img):
    # increase contrast
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

    return imgContrast


class DataLoader:
    """
    Loads all data from the file with the given path
    """

    def __init__(self, file_path, name):
        """
        :param file_path: The path of the dataset's directory.
        :param name: The name of the dataset.
        """

        self.samples = []

        with open(file_path + name) as f:
            bad_samples = []
            bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
            lines = f.readlines()

            for idx, line in enumerate(lines):
                # Remove the line break from the string:
                line = line.replace('\n', '')

                line_split = line.split(' ')
                line_split = list(filter(lambda x: x != '', line_split))
                assert len(line_split) >= 2

                file_name = file_path + line_split[0]
                gt_text = line_split[-1]

                # check if image is not empty
                if not os.path.getsize(file_name):
                    bad_samples.append(line_split[0] + '.png')
                    continue

                if len(line_split) == 3:
                    if line_split[1] not in {'ok', 'rw', 'er'}:
                        continue

                # put sample into list
                self.samples.append(Sample(gt_text, file_name))

                # display a loading bar to show loading progress.
                print('Loading "' + file_path + name + '": ' + display_loading_bar(idx, len(lines)), end='\r')

            print()
            print('Loaded {} samples.'.format(len(self.samples)))

            # some images in the IAM dataset are known to be damaged, don't show warning for them
            if set(bad_samples) != set(bad_samples_reference):
                print()
                print("Warning, damaged images found:", bad_samples)
                print("Damaged images expected:", bad_samples_reference)
