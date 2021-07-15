import sys
import colorama
from termcolor import colored

from library.utils import *
from library.const import *
from library.validate import validate
from library.evaluation import evaluate
from library.data.dataset_manager import load_dataset


def log_result(path, value):
    with open(path, 'w') as f:
        f.write(str(value))


def main():
    colorama.init()
    set_seeds()

    model = load_model(Const.baseline)
    ids = sys.argv[1:]

    for id in ids:
        print(colored('##########' + id + '##########', 'green'))

        if not os.path.isdir('../logs/experiment0/' + id):
            os.mkdir('../logs/experiment0/' + id)

        path = '../logs/experiment0/' + id + '/'

        domain_A_test = load_dataset(id + '-domain_A_test')
        domain_B_test = load_dataset(id + '-domain_B_test')
        domain_AB_test = domain_A_test + domain_B_test

        wer, cer = evaluate(validate(model, domain_A_test))
        log_result(path + 'test_A_cer.csv', cer)
        log_result(path + 'test_A_wer.csv', wer)

        wer, cer = evaluate(validate(model, domain_B_test))
        log_result(path + 'test_B_cer.csv', cer)
        log_result(path + 'test_B_wer.csv', wer)

        wer, cer = evaluate(validate(model, domain_AB_test))
        log_result(path + 'test_AB_cer.csv', cer)
        log_result(path + 'test_AB_wer.csv', wer)


if __name__ == '__main__':
    main()
