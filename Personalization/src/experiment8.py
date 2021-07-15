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


def adversarial(path, model, train_samples, iterations=10, size=100, num_folds=10):
    labels = [s.gt_text for s in train_samples]

    train_all = sorted(train_samples, key=lambda x: x.gt_text)
    random.seed(420)

    loss_deltas = dict()

    for i in range(iterations):
        print(colored('Iteration {}'.format(i), 'green'))
        random.shuffle(train_all)

        loss_deltas[i] = list()

        for step in range(0, size // 10 + 1):
            print(colored('Re-labeling {} out of {} samples.'.format(step * 10, size), 'blue'))

            train_set = train_all[:size]
            train_set = relabel(train_set, step * 10, labels)
            random.shuffle(train_set)

            loss_delta = 0

            for fold in range(num_folds):
                print('Train:', 0, 'to', fold*10, 'and', (fold+1)*10, 'to', size)
                print('Valid:', fold*10, 'to', (fold+1)*10)

                # Split the 100 training samples into a training and a validation set.
                train_split = train_set[0:fold*10] + train_set[(fold+1)*10:size]
                valid_split = train_set[fold*10:(fold+1)*10]
                valid_seq = to_valid_sequence(valid_split)

                # Measure the validation loss before and after retraining.
                init_loss = model.evaluate(x=valid_seq, verbose=True, steps=len(valid_seq))
                retrain(model, train_split, valid_split, verbose=True)
                fold_loss = model.evaluate(x=valid_seq, verbose=True, steps=len(valid_seq))
                loss_delta += (fold_loss - init_loss) / num_folds

                # Load the weights of the baseline model.
                print(colored('Loading weights', 'green'))
                model.load_checkpoint(Const.baseline)

            loss_deltas[i].append(loss_delta)

    pd.DataFrame.from_dict(loss_deltas, orient='index').to_csv(path + '/' + 'test_A_cer.csv')


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    model = load_model(Const.baseline)
    path = '../logs/experiment8/'

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

        adversarial(path + id, model, train_all)


if __name__ == '__main__':
    main()