from __future__ import division
from __future__ import print_function
from library.data.data_augmentation.transformation import affine_elastic

import numpy as np
import cv2

"""
Source: https://github.com/githubharald/SimpleHTR
"""

# initialize the pipeline once.
pipeline = affine_elastic()


def resize(img, tgt_size):
    """
    :param img: The image to resize.
    :param tgt_size: The target size for the image.
    :return: Returns the image with the target size.
    """

    # create target image and copy sample image into it
    (wt, ht) = tgt_size
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(wt, int(w / f)), 1),
                max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, new_size)
    target = np.ones([ht, wt]) * 255
    target[0:new_size[1], 0:new_size[0]] = img

    return target


def preprocess(img, img_size, augmenter=None):
    """put img into target img of size imgSize, transpose for TF and normalize gray-values"""
    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([img_size[1], img_size[0]])
        print('Not found!')

    # increase dataset size by applying random transformation to the image.
    if augmenter is not None:
        img = augmenter(image=img)

    # create target image and copy sample image into it
    img = resize(img, img_size)

    # transpose for TF
    img = cv2.transpose(img)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    # Cast the array to float 32 (only needed in branch coling)
    img = img.astype('float32')

    return np.expand_dims(img, axis=-1)
