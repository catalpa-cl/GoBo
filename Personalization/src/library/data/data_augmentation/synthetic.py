import os
import random
import cv2
import numpy as np

from PIL import ImageFont, ImageDraw, ImageOps, Image
from library.data.loader import Sample, load_lexicon
from library.data import display_loading_bar
from library.data.preprocessing import resize


class FilePaths:
    fpData = '../data/'
    fpWords = fpData + 'corpora/words.txt'
    fpFonts = fpData + 'fonts/ttf/'
    fpLog = '../logs/aug/'


def load_fonts(dir_path, glyph_height):
    """
    Loads all fonts from the given directory.

    :param dir_path: The directory to load the fonts from.
    :param glyph_height: The height of a glyph.
    :return: Returns a list of fonts.
    """
    fonts = list()

    for path in os.listdir(dir_path):
        fonts.append(ImageFont.truetype(dir_path + path, size=glyph_height))

    return fonts


def create_image(word, font, f_distrib, b_distrib):
    """
    Creates an image with the word written on it using the given font.

    :param word: The word to write.
    :param font: The font to use for writing the text.
    :param f_distrib:
    :param b_distrib:
    :return: Returns an image with synthetic handwriting.
    """
    f_mean, f_std_dev = f_distrib
    b_mean, b_std_dev = b_distrib
    w, h = font.getsize(word)

    # Make the image larger as the size returned by getsize is usually wrong.
    img = Image.new(mode='L', size=(2*w, 2*h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, h // 2), word, fill=int(f_mean + (255 - b_mean)), font=font)

    # Crop the image to only contain the text.
    bbox = ImageOps.invert(img).getbbox()
    img = img.crop(bbox)
    img = np.array(img)

    return img


def preprocess_image(img, img_size, f_distrib, b_distrib, blur_radius=5):
    """
    :param img: The image to preprocess.
    :param img_size: The image size.
    :param f_distrib: The distribution of the foreground pixels.
    :param b_distrib: The distribution of the background pixels.
    :param blur_radius: The radius for the blur.
    :return: Returns the preprocessed image.
    """

    f_mean, f_std_dev = f_distrib
    b_mean, b_std_dev = b_distrib

    # Add some noise to the image.
    img = img - (255 - b_mean)
    gauss = np.random.normal(0, (f_std_dev + b_std_dev) / 2, img.shape)
    img = np.add(img, gauss)

    # Add blur to the image. Currently not working as intended.
    img = cv2.GaussianBlur(img, (blur_radius, blur_radius), cv2.BORDER_DEFAULT)

    # Resize the image to the target size.
    img = resize(img, img_size)

    return img


def generate_images(samples_per_font, img_size, log=False):
    """
    Generate synthetic handwritten words.

    :param samples_per_font: The number of words per font.
    :param img_size: The size of the images.
    :param log: Write the results to the 'aug' folder in 'logs'.
    :return: Returns a list of samples.
    """

    samples = list()

    # Constant values from the IAM dataset:
    f_distrib = (92.30, 27.32)
    b_distrib = (243.79, 33.33)

    w, h = img_size

    words = set(load_lexicon(FilePaths.fpWords))
    fonts = load_fonts(FilePaths.fpFonts, 4*h)

    for i, font in enumerate(fonts):
        for j, word in enumerate(random.sample(words, k=samples_per_font)):
            img = create_image(word, font, f_distrib, b_distrib)

            # Some fonts don't support numbers which results in 0-dimensional images.
            while img.shape == ():
                word = random.sample(words, k=1)[-1]
                img = create_image(word, font, f_distrib, b_distrib)

            img = preprocess_image(img, img_size, f_distrib, b_distrib)
            samples.append(Sample(word, img=img))

            if log:
                cv2.imwrite(FilePaths.fpLog + word + '_' + str(i) + '_' + str(j) + '.png', img)

            # Display a progress bar as the generation might take longer.
            loading_bar = display_loading_bar(i * samples_per_font + j, len(fonts) * samples_per_font, verbose=True)
            print('Synthesizing:', loading_bar, end='\r')

    print()

    return samples


def main():
    generate_images(270, (256, 64), True)


if __name__ == '__main__':
    main()
