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


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    model = load_model(Const.baseline)

    if not os.path.isdir('../logs/experiment2/'):
        os.mkdir('../logs/experiment2/')

    domain_A_wers = {text: [] for text in texts + ['baseline']}
    domain_A_cers = {text: [] for text in texts + ['baseline']}
    domain_B_wers = {text: [] for text in texts + ['baseline']}
    domain_B_cers = {text: [] for text in texts + ['baseline']}
    domain_AB_wers = {text: [] for text in texts + ['baseline']}
    domain_AB_cers = {text: [] for text in texts + ['baseline']}

    for id in ids:
        print(colored('##########' + id + '##########', 'green'))

        domain_A_test = load_dataset(id + '-domain_A_test')
        domain_B_test = load_dataset(id + '-domain_B_test')
        domain_AB_test = domain_A_test + domain_B_test

        # Add the baseline model's performance on domain_A_test to the results.
        wer, cer = evaluate(validate(model, domain_A_test))
        domain_A_wers['baseline'].append(wer)
        domain_A_cers['baseline'].append(cer)

        # Add the baseline model's performance on domain_B_test to the results.
        wer, cer = evaluate(validate(model, domain_B_test))
        domain_B_wers['baseline'].append(wer)
        domain_B_cers['baseline'].append(cer)

        # Add the baseline model's performance on domain_AB_test to the results.
        wer, cer = evaluate(validate(model, domain_AB_test))
        domain_AB_wers['baseline'].append(wer)
        domain_AB_cers['baseline'].append(cer)

        for text in texts:
            print(colored('=> ' + text, 'red'))
            train_samples = load_dataset(id + '-' + text)

            retrain(model, train_samples, domain_AB_test)

            wer, cer = evaluate(validate(model, domain_A_test))
            domain_A_wers[text].append(wer)
            domain_A_cers[text].append(cer)

            wer, cer = evaluate(validate(model, domain_B_test))
            domain_B_wers[text].append(wer)
            domain_B_cers[text].append(cer)

            wer, cer = evaluate(validate(model, domain_AB_test))
            domain_AB_wers[text].append(wer)
            domain_AB_cers[text].append(cer)

            # Load the weights of the baseline model.
            print(colored('Loading weights', 'green'))
            model.load_checkpoint(Const.baseline)

    DataFrame(domain_A_wers).to_csv('../logs/experiment2/domain_A_wer.csv')
    DataFrame(domain_A_cers).to_csv('../logs/experiment2/domain_A_cer.csv')

    DataFrame(domain_B_wers).to_csv('../logs/experiment2/domain_B_wer.csv')
    DataFrame(domain_B_cers).to_csv('../logs/experiment2/domain_B_cer.csv')

    DataFrame(domain_AB_wers).to_csv('../logs/experiment2/domain_AB_wer.csv')
    DataFrame(domain_AB_cers).to_csv('../logs/experiment2/domain_AB_cer.csv')


if __name__ == '__main__':
    main()
