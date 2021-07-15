import colorama

from library.utils import *
from library.evaluation.evaluation import evaluate
from library.evaluation.errors import analyse_errors
from library.data.tokenizer import Tokenizer
from library.data.dataset_manager import request_datasets
from library.data.loader import get_labels


def validate(model, valid_samples):
    """
    Validates the model on the given dataset.
    :param model: The model to validate.
    :param valid_samples: The dataset to use for validation
    :return: Returns a list consisting of triples of shape (label, prediction, probability)
    """
    valid_seq = to_valid_sequence(valid_samples)
    valid_seq.predict = True

    labels = get_labels(valid_samples)

    # Use the model to predict the dataset
    predictions = model.predict_generator(
        generator=valid_seq,
        steps=len(valid_seq),
        verbose=1
    )

    # Create a tokenizer to decode the network's output
    tokenizer = Tokenizer(Const.charset)

    # Create a result list consisting of triples of shape (label, prediction, probability)
    results = [(label, tokenizer.decode(pred), prob) for label, (pred, prob) in zip(labels, predictions)]

    return results


def main():
    colorama.init()
    set_seeds()

    model = load_model(Const.baseline)

    # Ask the user to select a dataset.
    datasets = request_datasets(['validation'])
    valid_samples = datasets['validation']

    results = validate(model, valid_samples)

    log_path = '../logs/validation/'
    evaluate(results, log_path=log_path, verbose=True)
    analyse_errors(results, log_path=log_path)


if __name__ == '__main__':
    main()
