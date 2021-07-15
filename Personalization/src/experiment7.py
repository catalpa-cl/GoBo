import sys
import colorama
import pandas as pd

from termcolor import colored

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets
from library.data.loader import Sample


def relabel(samples, num_incorrect, labels):
    relabeled_set = list()

    for idx, sample in enumerate(samples):
        if idx >= num_incorrect:
            relabeled_set.append(sample)
        else:
            label = random.choice(labels)

            while label == sample.gt_text:
                label = random.choice(labels)

            print(sample.gt_text, '-->', label)

            relabeled_set.append(Sample(label, sample.img_path))

    return relabeled_set


def adversarial(path, model, train_samples, test_A_samples, test_B_samples, iterations=10, size=100):
    labels = [s.gt_text for s in train_samples]

    test_AB_samples = test_A_samples + test_B_samples

    train_all = sorted(train_samples, key=lambda x: x.gt_text)
    random.seed(420)

    test_A_wers = dict()
    test_A_cers = dict()

    test_B_wers = dict()
    test_B_cers = dict()

    test_AB_wers = dict()
    test_AB_cers = dict()

    base_A_wer, base_A_cer = evaluate(validate(model, test_A_samples))
    base_B_wer, base_B_cer = evaluate(validate(model, test_B_samples))
    base_AB_wer, base_AB_cer = evaluate(validate(model, test_AB_samples))

    for i in range(iterations):
        print(colored('Iteration {}'.format(i), 'green'))
        random.shuffle(train_all)

        test_A_wers[i] = [base_A_wer]
        test_A_cers[i] = [base_A_cer]

        test_B_wers[i] = [base_B_wer]
        test_B_cers[i] = [base_B_cer]

        test_AB_wers[i] = [base_AB_wer]
        test_AB_cers[i] = [base_AB_cer]

        for step in range(0, size // 10 + 1):
            print(colored('Re-labeling {} out of {} samples.'.format(step * 10, size), 'blue'))

            train_set = train_samples[:size]
            train_set = relabel(train_set, step * 10, labels)
            random.shuffle(train_set)

            retrain(model, train_set, test_A_samples, verbose=True)

            print(colored('Domain A:', 'red'))
            wer, cer = evaluate(validate(model, test_A_samples))
            test_A_cers[i].append(cer)
            test_A_wers[i].append(wer)

            print(colored('Domain B:', 'red'))
            wer, cer = evaluate(validate(model, test_B_samples))
            test_B_cers[i].append(cer)
            test_B_wers[i].append(wer)

            print(colored('Domain AB:', 'red'))
            wer, cer = evaluate(validate(model, test_AB_samples))
            test_AB_cers[i].append(cer)
            test_AB_wers[i].append(wer)

            # Load the weights of the baseline model.
            print(colored('Loading weights', 'green'))
            model.load_checkpoint(Const.baseline)

    pd.DataFrame.from_dict(test_A_cers, orient='index').to_csv(path + '/' + 'test_A_cer.csv')
    pd.DataFrame.from_dict(test_A_wers, orient='index').to_csv(path + '/' + 'test_A_wer.csv')

    pd.DataFrame.from_dict(test_B_cers, orient='index').to_csv(path + '/' + 'test_B_cer.csv')
    pd.DataFrame.from_dict(test_B_wers, orient='index').to_csv(path + '/' + 'test_B_wer.csv')

    pd.DataFrame.from_dict(test_AB_cers, orient='index').to_csv(path + '/' + 'test_AB_cer.csv')
    pd.DataFrame.from_dict(test_AB_wers, orient='index').to_csv(path + '/' + 'test_AB_wer.csv')


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)

    if not os.path.isdir('../logs/experiment7/'):
        os.mkdir('../logs/experiment7/')

    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    for id in ids:
        train_request = '+'.join([id + '-' + text for text in texts])
        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        # Load all necessary datasets.
        train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)

        if not os.path.isdir('../logs/experiment7/' + id):
            os.mkdir('../logs/experiment7/' + id)

        adversarial('../logs/experiment7/' + id, model, train_all, test_A, test_B)


if __name__ == '__main__':
    main()