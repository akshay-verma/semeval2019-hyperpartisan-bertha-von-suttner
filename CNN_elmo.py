import ast
import numpy as np
import tensorflow as tf

from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.layers import Input, Flatten, Dense, Activation
from keras.layers import Concatenate, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser


def load_elmo(path, max_len=200):
    '''
    load ELMo embedding from tsv file.
    :param path: tsv file path.
    :param to_pickle: Convert elmo embeddings to .npy file, avoid read and pad every time.
    :return: elmo embedding and its label.
    '''
    X = []
    label = []
    ids = []
    i = 0
    l_encoder = LabelEncoder()
    with open(path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            X.append(padded_seq)
            label.append(gzip_label)
            ids.append(gzip_id)
            i += 1
    Y = l_encoder.fit_transform(label)

    return np.array(X), np.array(Y), np.array(ids)


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(1e-7,
                                        dtype=output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))
    output = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                     logits=output)
    return K.mean(output, axis=-1)


def conv1d(max_len, embed_size):
    '''
    CNN without Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN without BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    drop = 0.5
    inputs = Input(shape=(max_len, embed_size), dtype='float32')

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(act_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(act_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(act_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(act_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(act_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    # model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model


def conv1d_BN(max_len, embed_size):
    '''
    CNN with Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN with BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    inputs = Input(shape=(max_len, embed_size), dtype='float32')
    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    bn_0 = BatchNormalization(momentum=0.7)(act_0)

    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    bn_1 = BatchNormalization(momentum=0.7)(act_1)

    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    bn_2 = BatchNormalization(momentum=0.7)(act_2)

    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    bn_3 = BatchNormalization(momentum=0.7)(act_3)

    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)
    bn_4 = BatchNormalization(momentum=0.7)(act_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(bn_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(bn_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(bn_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(bn_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(bn_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=1, activation='sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)
    # model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss=binary_crossentropy, metrics=['acc'], optimizer='adam')
    return model


def gen(data, target, batch_size):
    idx = 0
    while True:
        batchData, batchTarget = [], []
        while len(batchData) < batch_size:
            batchData.append(data[idx])
            batchTarget.append(data[idx])
            idx = (idx + 1) % len(data)


def ohemDataGenerator(model, datagen, batch_size):
    while True:
        samples, targets = [], []
        while len(samples) < batch_size:
            x_data, y_data = next(datagen)
            preds = model.predict(x_data)
            errors = np.abs(preds - y_data).max(axis=-1) > .99
            samples += x_data[errors].tolist()
            targets += y_data[errors].tolist()

        regular_samples = batch_size * 2 - len(samples)
        x_data, y_data = next(datagen)
        samples += x_data[:regular_samples].tolist()
        targets += y_data[:regular_samples].tolist()

        samples, targets = map(np.array, (samples, targets))

        idx = np.arange(batch_size * 2)
        np.random.shuffle(idx)
        batch1, batch2 = np.split(idx, 2)
        yield samples[batch1], targets[batch1]
        yield samples[batch2], targets[batch2]


def dataGenerator(data, target, batchSize, pos2negRatio=0.3):
    positiveId = np.where(target == 1)[0]
    negativeId = np.where(target == 0)[0]
    posNum = int(batchSize * pos2negRatio)
    posIdx = 0
    negIdx = 0
    finalIds = []
    while True:
        finalIds = []
        for _ in range(posNum):
            finalIds.append(positiveId[posIdx])
            posIdx = (posIdx + 1) % len(positiveId)
        for _ in range(batchSize - posNum):
            finalIds.append(negativeId[negIdx])
            negIdx = (negIdx + 1) % len(negativeId)
        finalIds = np.array(finalIds)
        np.random.shuffle(finalIds)
        yield data[finalIds], target[finalIds]


parser = ArgumentParser()
parser.add_argument("inputTSV", help="Elmo format input file")
args = parser.parse_args()

seed = 7
max_len = 200
embed_size = 1024
batch_size = 32

# Split training and validation data
x_data, y_data, ids = load_elmo(args.inputTSV, max_len=max_len)
trainData, valData, trainTarget, valTarget = train_test_split(x_data, y_data, random_state=seed)

# Create the model
model = conv1d_BN(max_len, embed_size)

# Callback
checkpoints = ModelCheckpoint(
    filepath='./saved_models/BNCNN_vacc{val_acc:.4f}_e{epoch:02d}.hdf5',
    verbose=1, monitor='val_acc', save_best_only=True)

# Train the model
# history = model.fit_generator(
#     dataGenerator(trainData, trainTarget, batch_size),
#     steps_per_epoch=len(trainData) // batch_size, epochs=30,
#     validation_data=dataGenerator(valData, valTarget, batch_size),
#     validation_steps=len(valData) // batch_size, callbacks=[checkpoints])

history = model.fit_generator(
    ohemDataGenerator(
        model, gen(trainData, trainTarget, batch_size), batch_size),
    steps_per_epoch=len(trainData) // batch_size, epochs=30,
    validation_data=ohemDataGenerator(
        model, gen(valData, valTarget, batch_size), batch_size),
    validation_steps=len(valData) // batch_size, callbacks=[checkpoints])
