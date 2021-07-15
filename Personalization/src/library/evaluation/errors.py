import pandas as pd
import numpy as np

from library.const import Const
from library.evaluation.alignment import compute_alignment


class ConfusionMatrix:
    def __init__(self):
        self.chars = list(Const.charset)
        self.mat = pd.DataFrame(columns=self.chars, index=self.chars, dtype='int64')
        self.mat.fillna(value=0, inplace=True)

    def add(self, l_c, p_c):
        self.mat.at[l_c, p_c] += 1

    def add_word(self, label, pred):
        for l_c, p_c in zip(label, pred):
            self.add(l_c, p_c)

    def __get_row_total(self, row):
        total = 0

        for col in self.chars:
            total += self.mat.at[row, col]

        return total

    def __get_col_total(self, col):
        total = 0

        for row in self.chars:
            total += self.mat.at[row, col]

        return total

    def get_statistics(self, log_path=None, verbose=False):
        stats = pd.DataFrame(columns=['total', 'err', 'precision', 'recall', 'f1'], index=self.chars, dtype=np.float)

        for c in self.chars:
            row_sum = self.__get_row_total(c)
            col_sum = self.__get_col_total(c)

            # Compute how many times a character wasn't recognized at all.
            stats.at[c, 'total'] = row_sum

            if col_sum != 0:
                precision = self.mat.at[c, c] / col_sum
            else:
                precision = 1

            if row_sum != 0:
                recall = self.mat.at[c, c] / row_sum
                err = self.mat.at[c, '#'] / row_sum
            else:
                recall = 1
                err = 0

            stats.at[c, 'err'] = err
            stats.at[c, 'precision'] = precision
            stats.at[c, 'recall'] = recall
            stats.at[c, 'f1'] = (2 * precision * recall) / (precision + recall)

        if verbose:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(stats)

        if log_path is not None:
            self.mat.to_csv(log_path + 'confusion_matrix.csv', sep=';')
            stats.to_csv(log_path + 'statistics.csv', sep=';')


def analyse_errors(results, log_path=None, verbose=False):
    conf_mat = ConfusionMatrix()

    for label, pred, prob in results:
        label, pred = compute_alignment(label, pred)

        conf_mat.add_word(label, pred)

    conf_mat.get_statistics(log_path=log_path, verbose=verbose)


def main():
    results = [
        ('Helloween', 'Hellon', 1),
        ('Namescreen', 'Nmsre', 1),
    ]

    analyse_errors(results, log_path='../../logs/validation/', verbose=True)


if __name__ == '__main__':
    main()
