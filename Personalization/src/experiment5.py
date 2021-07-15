import sys
import colorama
from termcolor import colored

from library.utils import *
from library.const import *
from library.data.dataset_manager import load_dataset

from experiment0 import compute_correlation


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    model = load_model(Const.baseline)

    # Create a folder for this experiment in the log directory.
    if not os.path.isdir('../logs/experiment5/'):
        os.mkdir('../logs/experiment5/')

    for id in ids:
        print(colored('################## {} ##################'.format(id), 'red'))

        # Create a folder for this id in the log directory.
        if not os.path.isdir('../logs/experiment5/' + id):
            os.mkdir('../logs/experiment5/' + id)

        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        domain_A_test = load_dataset(test_A_request)
        domain_B_test = load_dataset(test_B_request)

        for text in texts:
            print(colored('ID: {} Text: {}'.format(id, text), 'yellow'))

            log_path = '../logs/experiment5/' + id + '/' + text + '/'

            # Create a folder for the text in the id directory.
            if not os.path.isdir(log_path):
                os.mkdir(log_path)

            train_set = load_dataset(id + '-' + text)

            # Compute the correlation between samples used from the text and the model's performance.
            compute_correlation(log_path, model, train_set, domain_A_test, domain_B_test)


if __name__ == '__main__':
    main()
