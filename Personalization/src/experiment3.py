import sys

import colorama
from termcolor import colored
from pandas import DataFrame

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_dataset


def load_id(id):
    train_samples = load_dataset(id.join([id + '-brown+', '-cedar+', '-domain_A_train+', '-domain_B_train+', '-nonwords']))
    valid_samples = load_dataset(id.join([id + '-domain_A_test+', '-domain_B_test+']))

    return train_samples, valid_samples


def load_ids(ids):
    train_samples = []
    valid_samples = []

    for id in ids:
        train, valid = load_id(id)

        train_samples += train
        valid_samples += valid

    return train_samples, valid_samples


def main():
    colorama.init()
    set_seeds()

    model = load_model(Const.baseline)

    ids = set(os.listdir('../data/datasets/retrain/coling/')) - {'State.xlsx'}

    if not os.path.isdir('../logs/experiment3/'):
        os.mkdir('../logs/experiment3/')

    # Evaluate the baseline model in the IAM testset.
    iam_test = load_dataset('iam-test')
    # base_iam_wer, base_iam_cer = evaluate(validate(model, iam_test))

    for id in sys.argv[1:]:
        print(colored('##########' + id + '##########', 'green'))

        # Load the training samples for the given id
        train_id, valid_id = load_id(id)

        # Load the training samples for the other ids
        train_other, valid_other = load_ids(ids - {id})

        # Evaluate the baseline model on the id's validation set.
        # base_id_wer, base_id_cer = evaluate(validate(model, valid_id))

        # Evaluate the baseline model on the other id's validation set.
        # base_other_wer, base_other_cer = evaluate(validate(model, valid_other))

        # Reset the seed to make sure the shuffles are the same across all id.
        random.seed(420)
        """
        wers = {result: [] for result in ['base_id', 'retrain_id', 'base_iam',
                                          'retrain_iam', 'base_other', 'retrain_other']}
        cers = {result: [] for result in ['base_id', 'retrain_id', 'base_iam',
                                          'retrain_iam', 'base_other', 'retrain_other']}
        """

        wers = {'retrain_other': []}
        cers = {'retrain_other': []}

        for i in range(10):
            random.shuffle(train_id)

            # wers['base_id'].append(base_id_wer)
            # cers['base_id'].append(base_id_cer)
            # wers['base_iam'].append(base_iam_wer)
            # cers['base_iam'].append(base_iam_cer)
            # wers['base_other'].append(base_other_wer)
            # cers['base_other'].append(base_other_cer)

            # Retrain the model on the training data for the target id.
            retrain(model, train_id, valid_id)

            # iam_wer, iam_cer = evaluate(validate(model, iam_test))
            # id_wer, id_cer = evaluate(validate(model, valid_id))
            other_wer, other_cer = evaluate(validate(model, valid_other))

            # wers['retrain_id'].append(id_wer)
            # cers['retrain_id'].append(id_cer)
            # wers['retrain_iam'].append(iam_wer)
            # cers['retrain_iam'].append(iam_cer)
            wers['retrain_other'].append(other_wer)
            cers['retrain_other'].append(other_cer)

            # Reload the baseline weights to continue with the second id.
            model.load_checkpoint(Const.baseline)

        if not os.path.isdir('../logs/experiment3/' + id):
            os.mkdir('../logs/experiment3/' + id)

        DataFrame.from_dict(wers).to_csv('../logs/experiment3/' + id + '/wers.csv')
        DataFrame.from_dict(cers).to_csv('../logs/experiment3/' + id + '/cers.csv')


if __name__ == '__main__':
    main()
