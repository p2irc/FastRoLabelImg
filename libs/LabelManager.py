import numpy as np
import mmcv
import os

class LabelManager:
    def __init__(self, num_of_labels, filepath):
        if os.path.exists(filepath):
            data = mmcv.load(filepath)
            self._path = filepath
            self._num_of_labels = data['num_of_labels']
            self._placeholder_labels = data['placeholder_labels']
            self._mapping = data['mapping']
            self._labels = data['labels']
        else:
            self._path = filepath
            self._num_of_labels = num_of_labels
            self._placeholder_labels = self.generate_placeholder_labels()
            self._mapping = {}
            self._labels = []

            self.save_data()

    def save_data(self):
        data = {}
        data['num_of_labels'] = self._num_of_labels
        data['placeholder_labels'] = self._placeholder_labels
        data['mapping'] = self._mapping
        data['labels'] = self._labels

        mmcv.dump(data, self._path)

    def generate_placeholder_labels(self):
        labels = []
        for i in range(1, self._num_of_labels):
            labels.append("label_" + str(i))

        return tuple(labels)

    def add_label(self, label):
        if label not in self._labels:
            self._labels.append(label)
            ln = len(self._labels)
            self._mapping[self._placeholder_labels[ln - 1]] = label

    def get_placeholder_label(self, label, add=True):
        if label not in self._labels:
            if add:
                self.add_label(label)
            else:
                print("Not found")

        return list(self._mapping.keys())[list(self._mapping.values()).index(label)]

    def get_placeholder_labels(self):
        return self._placeholder_labels

    def get_label(self, placeholder_label):
        return self._mapping[placeholder_label]

    def print_labels(self):
        print(self._mapping)
