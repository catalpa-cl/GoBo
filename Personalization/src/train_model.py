import argparse

from library import *


def main():
    colorama.init()
    set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='', action='store_true')
    parser.add_argument('--synthetic', help='', action='store_true')
    parser.add_argument('--transformations', help='', action='store_true')
    parser.add_argument('--multi-gpu', help='', action='store_true')
    parser.add_argument('--train', nargs='?', const=1, type=str)
    parser.add_argument('--valid', nargs='?', const=1, type=str)
    parser.add_argument('--test', nargs='?', const=1, type=str)
    args = parser.parse_args()

    # Create a model and initialize its weights
    if args.load:
        model = load_model('../../logs/training/checkpoint.hdf5')
    else:
        model = load_model()

    # Ask the user to select the training and validation dataset.
    train_samples, valid_samples, test_samples = load_datasets(args.train, args.valid, args.test)

    print('Training: {} samples'.format(len(train_samples)))
    print('Validation: {} samples'.format(len(valid_samples)))
    print('Testing: {} samples'.format(len(test_samples)))

    # Train the model.
    train(model, train_samples, valid_samples, args.transformations)

    # Evaluate the resulting model
    evaluate(validate(model, test_samples), verbose=True)


if __name__ == '__main__':
    main()
