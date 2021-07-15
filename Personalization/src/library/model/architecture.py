from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, TimeDistributed, Attention
from tensorflow.keras.optimizers import RMSprop, Adam

"""
Source: 
https://medium.com/@arthurflor23/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16
https://github.com/arthurflor23/handwritten-text-recognition
"""


def create_conv_layer(input_layer, filters, kernel_size, pool_size, pool_strides, trainable=True, dropout=False):
    cnn = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), trainable=trainable)(input_layer)
    cnn = BatchNormalization(trainable=trainable)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    # Add a pooling layer if requested
    if pool_size != (1, 1) and pool_size != (1, 1):
        cnn = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding="valid")(cnn)

    if dropout:
        cnn = Dropout(rate=0.2, trainable=trainable)(cnn)

    return cnn


def custom_architecture(input_size, output_size, lr=3e-4, use_conv=True, use_lstm=True):
    """
    :param input_size: The size of the input to the network.
    :param output_size: The size of the output.
    :param lr: The initial learning rate.
    :param use_conv: Whether to set the convolutional layers to be trainable
    :param use_lstm: Whether to set the LSTM layers to be trainable
    :return: Returns the input, output and the optimizer.
    """

    input_data = Input(name="input", shape=input_size)

    cnn_block_1 = create_conv_layer(
        input_layer=input_data,
        filters=64,
        kernel_size=(5, 5),
        pool_size=(2, 2),
        pool_strides=(2, 2),
        trainable=use_conv
    )

    cnn_block_2 = create_conv_layer(
        input_layer=cnn_block_1,
        filters=64,
        kernel_size=(5, 5),
        pool_size=(2, 2),
        pool_strides=(2, 2),
        trainable=use_conv
    )

    cnn_block_3 = create_conv_layer(
        input_layer=cnn_block_2,
        filters=128,
        kernel_size=(5, 5),
        pool_size=(2, 2),
        pool_strides=(2, 2),
        trainable=use_conv
    )

    cnn_block_4 = create_conv_layer(
        input_layer=cnn_block_3,
        filters=128,
        kernel_size=(5, 5),
        pool_size=(1, 2),
        pool_strides=(1, 2),
        trainable=use_conv
    )

    print(cnn_block_4.shape)

    cnn_block_5 = create_conv_layer(
        input_layer=cnn_block_4,
        filters=256,
        kernel_size=(3, 3),
        pool_size=(1, 2),
        pool_strides=(1, 2),
        trainable=use_conv
    )

    cnn_block_6 = create_conv_layer(
        input_layer=cnn_block_5,
        filters=256,
        kernel_size=(3, 3),
        pool_size=(1, 2),
        pool_strides=(1, 2),
        trainable=use_conv,
        dropout=True
    )

    cnn_block_7 = create_conv_layer(
        input_layer=cnn_block_6,
        filters=512,
        kernel_size=(3, 3),
        pool_size=(1, 1),
        pool_strides=(1, 1),
        trainable=use_conv,
        dropout=True
    )

    shape = cnn_block_7.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn_block_7)

    # ====================== BLSTM 0 ======================

    blstm = Bidirectional(LSTM(units=512, return_sequences=True, trainable=use_lstm))(blstm)
    blstm = Activation('relu')(blstm)

    # ====================== BLSTM 1 ======================

    blstm = Bidirectional(LSTM(units=512, return_sequences=True, trainable=use_lstm))(blstm)
    blstm = Activation('relu')(blstm)
    blstm = Dropout(rate=0.5)(blstm)

    # ====================== Dense 0 ======================

    output_data = Dense(units=output_size, activation="softmax")(blstm)
    optimizer = Adam(learning_rate=lr)

    return input_data, output_data, optimizer


def attention(input_size, output_size, lr=3e-4, use_conv=True, use_lstm=True):
    """
    :param input_size: The size of the input to the network.
    :param output_size: The size of the output.
    :param lr: The initial learning rate.
    :param use_conv: Whether to set the convolutional layers to be trainable
    :param use_lstm: Whether to set the LSTM layers to be trainable
    :return: Returns the input, output and the optimizer.
    """

    input_data = Input(name="input", shape=input_size)

    # ====================== Conv 0 ======================

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", trainable=use_conv)(input_data)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 1 ======================

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 2 ======================

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 3 ======================

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # ====================== Conv 4 ======================

    cnn = Dropout(rate=0.2, trainable=use_conv)(cnn)
    cnn = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # ====================== Conv 5 ======================

    cnn = Dropout(rate=0.2, trainable=use_conv)(cnn)
    cnn = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # ====================== Conv 6 ======================

    cnn = Dropout(rate=0.2, trainable=use_lstm)(cnn)
    cnn = Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding="same", trainable=use_conv)(cnn)

    cnn = BatchNormalization(trainable=use_conv)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    # Shape: (batch, new_rows, new_cols, filters)
    print(cnn.get_shape())
    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # ====================== BLSTM 0 ======================

    blstm = Bidirectional(LSTM(units=512, return_sequences=True, trainable=use_lstm))(blstm)
    blstm = Activation('relu')(blstm)

    # ====================== BLSTM 1 ======================

    blstm = Bidirectional(LSTM(units=512, return_sequences=True, trainable=use_lstm))(blstm)
    blstm = Activation('relu')(blstm)
    blstm = Dropout(rate=0.5, trainable=use_lstm)(blstm)

    # ====================== Dense 0 ======================
    """
    query = Dense(units=64)(blstm)
    key = Dense(units=64)(blstm)
    value = Dense(units=64)(blstm)
    attention = Attention()([query, key, value])
    """

    output_data = Dense(units=output_size, activation="softmax")(blstm)
    optimizer = Adam(learning_rate=lr)

    return input_data, output_data, optimizer
