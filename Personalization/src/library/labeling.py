import colorama
from editdistance import eval


from library.utils import *
from library.validate import validate
from library.data.dataset_manager import request_datasets
from library.data.loader import samples_to_dataset


def main():
    colorama.init()
    set_seeds()

    model = load_model(Const.baseline)

    # Ask the user to select a dataset.
    datasets = request_datasets(['dataset'])
    samples = datasets['dataset']

    labels = [sample.gt_text for sample in samples]

    results = validate(model, samples)

    for sample, (_, pred, _) in zip(samples, results):
        distances = [(label, eval(label, pred)) for label in labels]
        closest = list(sorted(distances, key=lambda tup: tup[-1]))

        sample.gt_text = ' '.join([label for label, _ in closest[:5]])

    samples_to_dataset("relabeled.txt", samples)


if __name__ == '__main__':
    main()

