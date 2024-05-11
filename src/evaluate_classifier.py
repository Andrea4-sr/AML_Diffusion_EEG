import argparse
from collections import Counter
from eeg_preprocessing import EEGPreprocessor
import numpy
import os
import pathlib
import pickle
import pywt
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import torchvision


class _EEGPreprocessor:
    def __init__(self, sampling_rate, lowcut, highcut):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
    
    def __call__(self, data):
        preprocessor = EEGPreprocessor(data)
        return preprocessor.bandpass_fitler(data, self.sampling_rate, self.lowcut, self.highcut)


def _signal_to_features(signal):
     signal = signal.squeeze()
     std = numpy.std(signal)
     percentiles = numpy.percentile(signal, [10, 20, 30, 40, 50, 60, 70, 80, 90])
     return numpy.append(std, percentiles)


class EEGSignalToFeaturesDWT:
     def __init__(self, wavelet, mode):
          self.wavelet = wavelet
          self.mode = mode
     
     def __call__(self, signal):
          features = [_signal_to_features(n) for n in pywt.wavedec(signal, wavelet = self.wavelet, mode = self.mode)]
          return numpy.asarray(features).flatten()


class ClassifierPerformanceMetrics:
    def __init__(self, y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.tpr = tp / (tp + fn)
        self.tnr = tn / (tn + fp)
        self.f1_score = 2 * tp / (2 * tp + fp + fn)
        self.auroc = roc_auc_score(y_true, y_proba[:, 1])


def evaluate_classifier(model, dataset):
    x, y_true = zip(*dataset)
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)

    return ClassifierPerformanceMetrics(y_true, y_pred, y_proba)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluate a classifier model based on the provided dataset.')
    parser.add_argument('model_path', type = pathlib.Path, help = 'Path to the model dump file')
    parser.add_argument('dataset_path', type = pathlib.Path, help = 'Path to the dataset')

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f'Error: {args.model_path} does not exist (or is no file or not accessible)')
        quit()

    if not os.path.isdir(args.dataset_path):
        print(f'Error: {args.dataset_path} does not exist (or is no folder or not accessible)')
        quit()

    print(f'Loading model {args.model_path}')
    with open(args.model_path, 'rb') as file:
        model = pickle.load(file)

    print(f'Loading dataset {args.dataset_path}')
    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                 loader = lambda path: numpy.load(path),
                                                 extensions = ("npy"),
                                                 transform = torchvision.transforms.Compose([
                                                     numpy.squeeze,
                                                     _EEGPreprocessor(250, 0.5, 60),
                                                     EEGSignalToFeaturesDWT('db4', 'symmetric')
                                                 ]))
    
    print('')

    print(f'Number of samples in dataset: {len(dataset)}')
    print('Classes:')

    for target in Counter([t for _, t in dataset]).most_common():
        print(f'  {dataset.classes[target[0]]}: {target[1]}')
    
    print('')

    print('Evaluate classifier perfomance...')
    metrics = evaluate_classifier(model, dataset)

    print('')

    print(f'Accuracy: {metrics.accuracy}')
    print(f'TPR: {metrics.tpr}')
    print(f'TNR: {metrics.tnr}')
    print(f'F1 score: {metrics.f1_score}')
    print(f'AUROC: {metrics.auroc}')
    