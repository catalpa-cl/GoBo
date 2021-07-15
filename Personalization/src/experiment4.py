import sys
import colorama
from termcolor import colored

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation import evaluate
from library.data.dataset_manager import load_dataset
from library.data.loader import samples_to_dataset


def get_pred_gain(model, train_samples, valid_samples):
    pred_gains = list()

    valid_seq = to_valid_sequence(valid_samples)
    base_loss = model.evaluate(valid_seq)

    for sample in train_samples:
        retrain(model, [sample], valid_samples, epochs=1)
        pers_loss = model.evaluate(valid_seq)

        print(colored('{:.4f} -> {:.4f}'.format(base_loss, pers_loss), 'red'))

        pred_gains.append((sample, base_loss - pers_loss))
        model.load_checkpoint(Const.baseline)

    pred_gains = sorted(pred_gains, key=lambda tup: tup[-1])

    return pred_gains


def test_dataset(model, train_samples, valid_samples):
    retrain(model, train_samples, valid_samples)
    wer, cer = evaluate(validate(model, valid_samples))

    print(colored('{:.2f}% WER {:.2f}% CER'.format(wer, cer), 'red'))
    model.load_checkpoint(Const.baseline)


def print_prediction_gains(pred_gains):
    for sample, pred_gain in pred_gains:
        print('{} -> {:.4f}'.format(sample.gt_text, pred_gain))


def main():
    colorama.init()
    set_seeds()

    model = load_model(Const.baseline)

    ids = sys.argv[1:]

    for id in ids:
        train_set = load_dataset(id.join([id + '-brown+', '-cedar+', '-domain_A_train+', '-domain_B_train+', '-nonwords']))
        domain_A_test = load_dataset(id + '-domain_A_test')
        domain_B_test = load_dataset(id + '-domain_B_test')
        domain_AB_test = domain_A_test + domain_B_test

        pred_gains = get_pred_gain(model, train_set, domain_AB_test)
        best_samples = [sample for sample, _ in pred_gains[-100:]]
        worst_samples = [sample for sample, _ in pred_gains[:100]]

        samples_to_dataset(str(id) + '_pg_best.txt', best_samples)
        samples_to_dataset(str(id) + '_pg_worst.txt', worst_samples)

        print(colored('Best:', 'red'))
        test_dataset(model, best_samples, domain_AB_test)

        print(colored('Worst:', 'red'))
        test_dataset(model, worst_samples, domain_AB_test)


if __name__ == '__main__':
    main()