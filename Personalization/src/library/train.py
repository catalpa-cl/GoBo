from library import *


def train(model, train_samples, valid_samples, transformations=False):
    """
    :param model: The model to train.
    :param train_samples: The dataset used for training.
    :param valid_samples: The dataset used for validation.
    :param test_samples: The dataset used for testing.
    :param synthetic: Whether or not to add synthetic training examples to the dataset.
    :param transformations: Whether or not to apply random transformations to the dataset.
    """
    # Make sure that both the training and validation set are disjoint.
    if set(train_samples) & set(valid_samples):
        print(colored('The training and validation sets are not disjoint!', 'yellow'))
        exit()

    # Set up the data augmentation pipeline if requested.
    if transformations:
        augmenter = affine_elastic()
    else:
        augmenter = None

    # Convert both datasets into ImageSequences.
    train_seq, valid_seq = to_sequences(train_samples, valid_samples, augmenter)

    # Get the trainable CTC Model.
    model_train = model.model_train
    optimizer = model.model_train.optimizer
    model_pred = model.model_pred

    num_batches = len(train_seq)

    last_improved = 0
    best_val_loss = np.Inf
    es_patience = Const.train_settings.es_patience
    lr_patience = Const.train_settings.lr_patience
    lr_factor = Const.train_settings.lr_factor

    for epoch in range(Const.epochs):
        print('Start of epoch', epoch)

        model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            validation_data=valid_seq,
            validation_freq=10,
            epochs=1,
            shuffle=True,
            verbose=1
        )

        # Compute the training loop.
        val_loss = model.evaluate(
            x=valid_seq,
            verbose=True,
            steps=len(valid_seq),
        )

        print('Epoch', epoch, '-', 'val_loss={:.2f}'.format(val_loss))

        if val_loss < best_val_loss:
            print('Validation loss improved from {} to {}.'.format(best_val_loss, val_loss))

            last_improved = epoch
            best_val_loss = val_loss

            model.save_checkpoint('../logs/checkpoint.hdf5')
        else:
            # Early Stopping implementation.
            if epoch - last_improved > es_patience:
                print('EarlyStopping: Stopping training.')
                break

            # Reduce LR on plateau implementation.
            if epoch - last_improved > lr_patience:
                old_lr = optimizer.learning_rate
                new_lr = optimizer.learning_rate * lr_factor
                optimizer.learning_rate = new_lr

                print('ReduceLROnPlateau: Reducing lr from {} to {}.'.format(old_lr, new_lr))

    # Update the weights of the prediction model to the newly trained weights.
    model.load_checkpoint('../../logs/checkpoint.hdf5')
