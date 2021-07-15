import sys
import colorama
import random
import pandas as pd

from termcolor import colored

from library.utils import *
from library.const import *
from library.data.dataset_manager import load_datasets, load_dataset
from library.retrain import retrain
from library.retrain import retrain
from library.data.loader import Sample
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets


def load_samples(path):
    samples = list()

    with open(path) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            linesplit = line.split(',')
            samples.append(Sample(linesplit[1], linesplit[0]))

    return samples


def main():
    colorama.init()
    set_seeds()
    model = load_model(Const.baseline)

    ranking_dir = 'word_rankings/'
    dataset_sizes = [20, 40]
    datasets = ['A', 'B', 'AB']

    if not os.path.isdir('../logs/experiment11/'):
        os.mkdir('../logs/experiment11/')

    for dataset_size in dataset_sizes:
        for dataset in datasets:
            results = dict()

            for id in os.listdir(ranking_dir):
                results[id] = dict()

                test_A_request = id + '-' + 'domain_A_test'
                test_B_request = id + '-' + 'domain_B_test'
                testA = load_dataset(test_A_request)
                testB = load_dataset(test_B_request)
                testAB = testA + testB

                if dataset == 'A':
                    test_set = testA
                elif dataset == 'B':
                    test_set = testB
                else:
                    test_set = testAB

                wer, cer = evaluate(validate(model, test_set))
                results[id]['base_wer'] = wer
                results[id]['base_cer'] = cer

                samples = load_samples(ranking_dir + id + '/{}_{}.csv'.format(dataset, id))
                best = samples[:dataset_size]
                worst = samples[-dataset_size:]

                retrain(model, best, test_set)
                wer, cer = evaluate(validate(model, test_set))
                results[id]['best_wer'] = wer
                results[id]['best_cer'] = cer

                # Load the weights of the baseline model.
                print(colored('Loading weights', 'green'))
                model.load_checkpoint(Const.baseline)

                retrain(model, worst, test_set)
                wer, cer = evaluate(validate(model, test_set))
                results[id]['worst_wer'] = wer
                results[id]['worst_cer'] = cer

                # Load the weights of the baseline model.
                print(colored('Loading weights', 'green'))
                model.load_checkpoint(Const.baseline)

            pd.DataFrame.from_dict(results, orient='index').to_csv('../logs/experiment11/results_{}_{}.csv'.format(dataset, dataset_size))


if __name__ == '__main__':
    main()
