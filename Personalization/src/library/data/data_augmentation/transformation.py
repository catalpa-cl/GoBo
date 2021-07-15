import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def sometimes(aug):
    return iaa.Sometimes(0.7, aug)


def affine_elastic():
    """
    Defines a data augmentation pipelines that randomly adds transformations
    to the given image.

    Similar to:
    Dutta, Kartik, et al. "Improving cnn-rnn hybrid networks for handwriting recognition."
    2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR). IEEE, 2018.

    :return: Returns the pipeline as a callable function.
    """

    seq = iaa.Sequential([
        sometimes(
            iaa.Affine(
                translate_px=(-20, 20),
                scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                rotate=(-5, 5),
                shear=(-0.5, 0.5),
                order=[0, 1],
                cval=(243-30, 243+30),
                mode='constant',
                fit_output=True
            )
        ),
        sometimes(
            iaa.ElasticTransformation(
                alpha=(80, 100.0),
                sigma=(8, 10),
                order=[0, 1],
                cval=255,
                mode='constant'
            )
        )
    ])

    return seq


def affine():
    """
    Defines a data augmentation pipeline that only applies affine transformations to the images.

    :return: Returns the pipeline as a callable function.
    """
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.7,
            iaa.Affine(
                translate_px=(-20, 20),
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                order=[0, 1],
                cval=255,
                mode='constant',
                fit_output=True
            )
        )
    ])

    return seq
