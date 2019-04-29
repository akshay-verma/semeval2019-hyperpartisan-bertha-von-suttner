import ast
import argparse
import numpy as np

from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(data_path, max_len=200):
    data = []
    l = []
    ids = []
    i = 0
    l_encoder = LabelEncoder()
    with open(data_path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            data.append(padded_seq)
            l.append(gzip_label)
            ids.append(gzip_id)
            i += 1
    label = l_encoder.fit_transform(l)
    return np.array(data), np.array(label), np.array(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputTSV", help="Location of input data tsv")
    parser.add_argument("--savedModel", help="Location of saved model file")
    args, _ = parser.parse_known_args()

    maxLen = 200
    data, target, docId = load_data(args.inputTSV, max_len=maxLen)
    model = load_model(args.savedModel)
    pred = model.predict(data)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    print("Accuracy: {}".format(accuracy_score(target, pred)))
    print("F1 score: {}".format(f1_score(target, pred, average="macro")))
    print("Precision: {}".format(precision_score(target, pred, average="macro")))
    print("Recall: {}".format(recall_score(target, pred, average="macro")))
