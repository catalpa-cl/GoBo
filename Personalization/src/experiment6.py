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


def log_preds(model, test_samples, log_path):
    predictions = validate(model, test_samples)
    results = list()

    for sample, (label, pred, prob) in zip(test_samples, predictions):
        err = eval(pred, label)

        results.append([sample.img_path, label, pred, prob, err])

    DataFrame(data=np.asarray(results)).to_csv(log_path)

    return evaluate(predictions)


def log_words(path, train_samples):
    with open(path, 'w') as f:
        for sample in train_samples:
            f.write(sample.img_path + ', ' + sample.gt_text + '\n')


def compute_results(path, model, train_all, test_A, test_B, num_iterations=10, step_size=10):
    train_all = sorted(train_all, key=lambda x: x.gt_text)
    random.seed(420)

    test_AB = test_A + test_B

    # Compute the initial wer and cer on the three test sets.
    base_A_wer, base_A_cer = evaluate(validate(model, test_A))
    base_B_wer, base_B_cer = evaluate(validate(model, test_B))
    base_AB_wer, base_AB_cer = evaluate(validate(model, test_A + test_B))

    for i in range(num_iterations):
        random.shuffle(train_all)

        for step in range(1, len(train_all) // step_size):
            # Take all samples from step 1 to step 2.
            train_samples = train_all[(step-1)*step_size:step*step_size]

            print(colored('Retraining: {} to {}'.format((step-1)*step_size, step*step_size), 'green'))
            retrain(model, train_samples, test_A, verbose=True)

            word_log_path = path + '/preds_{}_{}_{}.csv'

            print(colored('Domain A:', 'red'))
            test_A_wer, test_A_cer = log_preds(model, test_A, log_path=word_log_path.format(i, step, 'A'))

            print(colored('Domain B:', 'red'))
            test_B_wer, test_B_cer = log_preds(model, test_B, log_path=word_log_path.format(i, step, 'B'))

            print(colored('Domain AB:', 'red'))
            test_AB_wer, test_AB_cer = log_preds(model, test_AB, log_path=word_log_path.format(i, step, 'AB'))

            log_words(path + '/paths_{}_{}.txt'.format(i, step), train_samples)

            results = {'A':  {'base_wer': base_A_wer,  'base_cer': base_A_cer,  'wer': test_A_wer,  'cer': test_A_cer},
                       'B':  {'base_wer': base_B_wer,  'base_cer': base_B_cer,  'wer': test_B_wer,  'cer': test_B_cer},
                       'AB': {'base_wer': base_AB_wer, 'base_cer': base_AB_cer, 'wer': test_AB_wer, 'cer': test_AB_cer}}

            DataFrame.from_dict(data=results, orient='index').to_csv(path + '/results_{}_{}.csv'.format(i, step))

            # Load the weights of the baseline model.
            print(colored('Loading weights', 'green'))
            model.load_checkpoint(Const.baseline)


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)

    if not os.path.isdir('../logs/experiment6/'):
        os.mkdir('../logs/experiment6/')

    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    for id in ids:
        train_request = '+'.join([id + '-' + text for text in texts])
        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        # Load all necessary datasets.
        train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)

        if not os.path.isdir('../logs/experiment6/' + id):
            os.mkdir('../logs/experiment6/' + id)

        compute_results('../logs/experiment6/' + id, model, train_all, test_A, test_B)


if __name__ == '__main__':
    main()
