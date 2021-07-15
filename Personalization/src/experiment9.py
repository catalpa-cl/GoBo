import sys
import colorama
import random
import pandas as pd

from termcolor import colored

from library.utils import *
from library.const import *
from library.data.dataset_manager import load_datasets
from library.retrain import retrain
from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets


def load_datasets_for_id(id):
    """
    Load the datasets (training and validation sets) for a given id.

    :param id: The id of the datasets.
    :return: Returns training set and test sets for A and B.
    """
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']
    train_request = '+'.join([id + '-' + text for text in texts])
    test_A_request = id + '-' + 'domain_A_test'
    test_B_request = id + '-' + 'domain_B_test'

    # Load all necessary datasets.
    train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)
    return train_all, test_A, test_B


def load_word_list(path):
    words = list()

    with open(path) as f:
        for line in f.readlines():
            words.append(line.replace('\n', ''))

    return words


def extract_words(samples, words):
    dataset = list()
    sample_dict = dict()

    for sample in samples:
        if sample.gt_text not in sample_dict:
            sample_dict[sample.gt_text] = list()

        sample_dict[sample.gt_text].append(sample)

    for word in words:
        sample = random.choice(sample_dict[word])

        dataset.append(sample)

    assert len(dataset) == len(words)

    return dataset


def test_word_lists(model, best, worst, train_all, test_A, test_B, path, num_iterations=10):
    test_AB = test_A + test_B
    results = dict()

    for i in range(num_iterations):
        print(colored('Iteration {}'.format(i), 'red'))

        best_set = extract_words(train_all, best)
        worst_set = extract_words(train_all, worst)

        init_A_wer, init_A_cer = evaluate(validate(model, test_A))
        init_B_wer, init_B_cer = evaluate(validate(model, test_B))
        init_AB_wer, init_AB_cer = evaluate(validate(model, test_AB))

        print(colored('Initial: {:.2f}% CER {:.2f}% WER'.format(init_AB_cer, init_AB_wer)))

        retrain(model, best_set, test_A, verbose=True)
        best_A_wer, best_A_cer = evaluate(validate(model, test_A))
        best_B_wer, best_B_cer = evaluate(validate(model, test_B))
        best_AB_wer, best_AB_cer = evaluate(validate(model, test_AB))

        print(colored('Best: {:.2f}% CER {:.2f}% WER'.format(best_AB_cer, best_AB_wer)))

        # Load the weights of the baseline model.
        print(colored('Loading weights', 'green'))
        model.load_checkpoint(Const.baseline)

        retrain(model, worst_set, test_A, verbose=True)
        worst_A_wer, worst_A_cer = evaluate(validate(model, test_A))
        worst_B_wer, worst_B_cer = evaluate(validate(model, test_B))
        worst_AB_wer, worst_AB_cer = evaluate(validate(model, test_AB))

        print(colored('Worst: {:.2f}% CER {:.2f}% WER'.format(worst_AB_cer, worst_AB_wer)))

        # Load the weights of the baseline model.
        print(colored('Loading weights', 'green'))
        model.load_checkpoint(Const.baseline)

        results[i] = {
            'init_A_wer': init_A_wer, 'init_A_cer': init_A_cer,
            'init_B_wer': init_B_wer, 'init_B_cer': init_B_cer,
            'init_AB_wer': init_AB_wer, 'init_AB_cer': init_AB_cer,
            'worst_A_wer': worst_A_wer, 'worst_A_cer': worst_A_cer,
            'worst_B_wer': worst_B_wer, 'worst_B_cer': worst_B_cer,
            'worst_AB_wer': worst_AB_wer, 'worst_AB_cer': worst_AB_cer,
            'best_A_wer': best_A_wer, 'best_A_cer': best_A_cer,
            'best_B_wer': best_B_wer, 'best_B_cer': best_B_cer,
            'best_AB_wer': best_AB_wer, 'best_AB_cer': best_AB_cer
        }

    pd.DataFrame.from_dict(data=results, orient='index').to_csv(path + '/results.csv')


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)
    path = '../logs/experiment9/'

    if not os.path.isdir(path):
        os.mkdir(path)

    worst_samples = load_word_list('worst.txt')
    best_samples = load_word_list('best.txt')

    for id in ids:
        train_all, test_A, test_B = load_datasets_for_id(id)

        if not os.path.isdir(path + id):
            os.mkdir(path + id)

        test_word_lists(model, best_samples, worst_samples, train_all, test_A, test_B, path + "{}/".format(id))


if __name__ == '__main__':
    main()
