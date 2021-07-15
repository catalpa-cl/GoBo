import sys
import colorama
import json

from termcolor import colored

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets


def log_results(path, samples, wer, cer):
    results = {
        'samples': [{'path': sample.img_path, 'label': sample.gt_text} for sample in samples],
        'wer': wer,
        'cer': cer
    }

    with open(path, 'w') as f:
        json.dump(results, f)


def compute_results(path, model, train_all, test_A, test_B, num_iterations=1000):
    set_size_A = 20
    set_size_B = 40

    train_all = sorted(train_all, key=lambda x: x.gt_text)
    random.seed(420)

    for i in range(num_iterations):
        train_A = random.sample(train_all, set_size_A)
        train_B = random.sample(train_all, set_size_B)

        if i < 100:
            continue

        print(i)

        # Retrain the model on train set A.
        retrain(model, train_A, test_A, verbose=False)
        wer, cer = evaluate(validate(model, test_A))

        # Log the results including the samples as a json file.
        log_results(path + '{}_A.json'.format(i), train_A, wer, cer)

        # Load the weights of the baseline model.
        print(colored('Loading weights', 'green'))
        model.load_checkpoint(Const.baseline)

        # Retrain the model on train set A.
        retrain(model, train_B, test_B, verbose=False)
        wer, cer = evaluate(validate(model, test_B))

        # Log the results including the samples as a json file.
        log_results(path + '{}_B.json'.format(i), train_B, wer, cer)

        # Load the weights of the baseline model.
        print(colored('Loading weights', 'green'))
        model.load_checkpoint(Const.baseline)


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)
    path = '../logs/experiment12/'

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

        compute_results(path + id + '/', model, train_all, test_A, test_B)


if __name__ == '__main__':
    main()
