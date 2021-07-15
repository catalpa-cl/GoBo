import sys
import colorama
import tqdm

from library.utils import *
from library.const import *
from library.validate import *

from library.data.dataset_manager import load_datasets


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    model = load_model(Const.baseline)

    with open('results.csv', 'w') as f:
        for id in ids:
            # The requests for the datasets for writer A.
            train_request = '+'.join([id + '-' + text for text in texts])
            test_A_request = id + '-' + 'domain_A_test'
            test_B_request = id + '-' + 'domain_B_test'

            # Load all necessary datasets.
            train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)
            all_samples = train_all + test_A + test_B

            results = validate(model, all_samples)

            for (label, pred, prob), sample in tqdm.tqdm(zip(results, all_samples)):

                f.write('{};{};{};{};{:.4f}\n'.format(id, sample.img_path, label, pred, prob))


if __name__ == '__main__':
    main()
