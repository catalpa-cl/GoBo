import colorama
from termcolor import colored
import time

from library.utils import *
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.evaluation.logger import Logger
from library.data.dataset_manager import request_datasets
from library.data.sequence import ImageSequence


def log_loss(path, loss, val_loss):
    """
    Logs the loss and validation loss to a log file.
    :param path: The path of the log file.
    :param loss: The trainings loss after each epoch.
    :param val_loss: The validation loss after each epoch.
    """
    logger = Logger()
    logger.add_file(open(path, 'w'))

    logger.log('loss', *loss, sep=',')
    logger.log('val_loss', *val_loss, sep=',')

    logger.close()


def retrain(model, train_samples, valid_samples, epochs=3, verbose=True):
    """
    Retrains the given model on the personalization set.
    :param model: The model to retrain.
    :param train_samples: The training set to use for personalization.
    :param valid_samples: The validation set to use during retraining.
    :param epochs: The number of epochs to retrain the model.
    :param verbose: Whether or not to print the training process to the console.
    :return: Returns the history of the training process.
    """

    # Create two sequences for the training and validation samples.
    if type(train_samples) is not ImageSequence and type(valid_samples) is not ImageSequence:
        # Make sure that both the training and validation set are disjoint.
        if set(train_samples) & set(valid_samples):
            print(colored('The training, validation and test sets are not disjoint!', 'yellow'))
            exit()

        train_seq, valid_seq = to_sequences(train_samples, valid_samples)
    else:
        train_seq = train_samples
        valid_seq = valid_samples

    if verbose:
        # Compute the training loop.
        val_loss = model.evaluate(
            x=valid_seq,
            verbose=True,
            steps=len(valid_seq),
        )

        print('Initial validation loss: {}'.format(val_loss))

    for epoch in range(epochs):
        if verbose:
            print('Start of epoch', epoch)

        model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            validation_data=valid_seq,
            validation_freq=10,
            epochs=1,
            verbose=verbose
        )

        val_loss = model.evaluate(
            x=valid_seq,
            verbose=verbose,
            steps=len(valid_seq),
        )

        if verbose:
            print('Validation loss: {}'.format(val_loss))


def run_retraining(model, train_samples, valid_samples, test_samples):
    """
    Retrains the model using the datasets and evaluates the model's performance before and after retraining.
    :param model: The model to retrain.
    :param train_samples: The training samples for retraining.
    :param valid_samples: The validation samples for retraining.
    :param test_samples: The test samples to evaluate the retrained model.
    """

    # Make sure that both the training and validation set are disjoint.
    if set(train_samples) & set(valid_samples) & set(test_samples):
        print(colored('The training, validation and test sets are not disjoint!', 'yellow'))
        exit()

    # File paths for logging.
    log_path = '../logs/retraining/'
    log_results_path = log_path + 'retrain_log_' + time.strftime("%H-%M-%S", time.localtime()) + '.txt'
    log_loss_path = log_path + 'retrain_log_loss_' + time.strftime("%H-%M-%S", time.localtime()) + '.txt'

    # Evaluate the pretrained model.
    evaluate(validate(model, test_samples))

    # Retrain the model
    retrain(model, train_samples, valid_samples)

    # Evaluate the resulting model:
    print('Error rates (validation):')
    evaluate(validate(model, valid_samples))

    print('Error rates (test):')
    evaluate(validate(model, test_samples), verbose=True, log_path=log_results_path)


def main():
    colorama.init()
    set_seeds()

    # Create a model and initialize its weights
    model = load_model(Const.baseline, use_conv=True)

    # Ask the user to select the training and validation dataset.
    datasets = request_datasets(['training', 'validation', 'test'])
    train_samples = datasets['training']
    valid_samples = datasets['validation']
    test_samples = datasets['test']

    run_retraining(model, train_samples, valid_samples, test_samples)


if __name__ == '__main__':
    main()




