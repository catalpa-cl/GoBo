import sys
import re
import colorama
import pandas as pd

from termcolor import colored
from pandas import DataFrame
from editdistance import eval

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets


def log_words(path, train_samples):
    with open(path, 'w') as f:
        for sample in train_samples:
            f.write(sample.img_path + ', ' + sample.gt_text + '\n')


def compute_results(path, model, train_all, test_A, test_B):
    train_all = sorted(train_all, key=lambda x: x.gt_text)
    random.seed(420)

    test_AB = test_A + test_B
    test_A_seq = to_valid_sequence(test_A)
    test_B_seq = to_valid_sequence(test_B)
    test_AB_seq = to_valid_sequence(test_AB)

    # Compute the initial loss on the three test sets.
    base_A_loss = model.evaluate(test_A_seq)
    base_B_loss = model.evaluate(test_B_seq)
    base_AB_loss = model.evaluate(test_AB_seq)

    for idx, sample in enumerate(train_all):
        print(colored(sample, 'red'))

        train_seq = ImageSequence(
            data=[sample],
            batch_size=1,
            img_size=Const.img_size,
            charset=Const.charset,
            max_word_len=Const.max_text_length,
        )

        valid_seq = to_valid_sequence(test_A[:1])

        retrain(model, train_seq, valid_seq, epochs=1, verbose=True)

        word_log_path = path + '/samples_{}.csv'
        log_words(word_log_path.format(idx), [sample])

        print(colored('Domain A:', 'red'))
        test_A_loss = model.evaluate(test_A_seq)

        print(colored('Domain B:', 'red'))
        test_B_loss = model.evaluate(test_B_seq)

        print(colored('Domain AB:', 'red'))
        test_AB_loss = model.evaluate(test_AB_seq)

        results = {'A': test_A_loss - base_A_loss, 'B': test_B_loss - base_B_loss, 'AB': test_AB_loss - base_AB_loss}

        DataFrame.from_dict(data=results, orient='index').to_csv(path + '/results_{}.csv'.format(idx))

        # Load the weights of the baseline model.
        print(colored('Loading weights', 'green'))
        model.load_checkpoint(Const.baseline)


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)

    path = '../logs/experiment6_pg/'

    if not os.path.isdir(path):
        os.mkdir(path)

    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    for id in ids:
        train_request = '+'.join([id + '-' + text for text in texts])
        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        # Load all necessary datasets.
        train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)

        if not os.path.isdir(path + id):
            os.mkdir(path + id)

        compute_results(path + id, model, train_all, test_A, test_B)


if __name__ == '__main__':
    main()
