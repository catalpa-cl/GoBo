import numpy as np
import unicodedata

"""
Source: 
https://medium.com/@arthurflor23/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16
https://github.com/arthurflor23/handwritten-text-recognition
"""


class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=32):
        """
        :param chars: The characters that can occur in the text.
        :param max_text_length: The maximum length of a text
        """

        self.chars = chars
        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length
        self.blank_label = len(chars)

    def encode(self, text):
        """
        Encode text to vector.

        :param text: The text to encode.
        :return: Returns the encoded text.
        """

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())
        encoded = []

        for item in text:
            index = self.chars.find(item)
            encoded.append(index)

        return np.array(encoded)

    def decode(self, text):
        """
        Decode vector to text.

        :param text: The vector to decode.
        :return: Returns the decoded text.
        """

        decoded = "".join([self.chars[int(x)] for x in text if int(x) != self.blank_label])

        return decoded
