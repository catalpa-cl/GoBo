from library.model.callbacks import CallbackSettings


class Const:
    baseline = '../checkpoints/coling.hdf5'

    # constant parameters for the input.
    input_size = (256, 64, 1)
    img_size = (256, 64)
    batch_size = 10

    # constant parameters for the output.
    charset = ' !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    max_text_length = 32
    output_size = len(charset)+1

    # constant parameters for the training.
    epochs = 100000
    val_split = 0.95
    learning_rate = 0.001

    retrain_settings = CallbackSettings(
        log_path='../logs/retraining/',
        hdf_name='retrained.hdf5',
        lr_fac=0.1,
        lr_pat=5,
        es_pat=10
    )

    train_settings = CallbackSettings(
        log_path='../logs/training/',
        hdf_name='checkpoint.hdf5',
        lr_fac=0.1,
        lr_pat=2,
        es_pat=2
    )
