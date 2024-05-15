import argparse
from collections import Counter
from eeg_preprocessing import EEGPreprocessor
import numpy
import numpy as np
import os
import pathlib
import pickle
import pywt
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.signal import welch
import torchvision
import itertools
import matplotlib.pyplot as plt


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

class EEGSignalToFeaturesFFT:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, signal):
        fft_result = np.fft.fft(signal)
        magnitude_spectrum = numpy.abs(fft_result)
        phase_spectrum = numpy.angle(fft_result)
        features = np.concatenate((magnitude_spectrum, phase_spectrum))
        return features

class EEGSignalToFeaturesCWT:

    def __init__(self, wavelet):
        self.wavelet = wavelet
        self.alpha_range = (8, 12)
        self.beta_range = (12, 30)
    def _calculate_medium_scales(self, sampling_rate):
        alpha_scales = pywt.scale2frequency(self.wavelet, np.arange(64, sampling_rate)) * sampling_rate
        beta_scales = pywt.scale2frequency(self.wavelet, np.arange(64, sampling_rate)) * sampling_rate

        alpha_medium_scales = alpha_scales[(alpha_scales >= self.alpha_range[0]) & (alpha_scales <= self.alpha_range[1])]
        beta_medium_scales = beta_scales[(beta_scales >= self.beta_range[0]) & (beta_scales <= self.beta_range[1])]

        medium_scales = np.concatenate((alpha_medium_scales, beta_medium_scales))
        return medium_scales

    def __call__(self, signal):
        medium_scales = self._calculate_medium_scales(sampling_rate=250)

        features = []
        for scale in medium_scales:
            coefficients, _ = pywt.cwt(signal, scales=[scale], wavelet=self.wavelet)
            features.append(coefficients.flatten())
        features_array = np.asarray(features)
        print(features_array.shape)
        exit()
        return np.asarray(features).flatten()


class EEGSignalToFeaturesWelch:
    def __init__(self, sampling_rate, nperseg=None):
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg or sampling_rate // 2

    def __call__(self, signal):
        frequency, power = welch(signal, fs=self.sampling_rate, nperseg=self.nperseg)
        power = power[:70]
        power = np.log(power)
        return power



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
    parser.add_argument('--model_path', type = pathlib.Path, help = 'Path to the model dump file')
    parser.add_argument('--dataset_path', type = pathlib.Path, help = 'Path to the dataset')
    parser.add_argument('--plot_path', type=pathlib.Path, help = 'Path to dump the plot')
    parser.add_argument('--metrics_path', type = pathlib.Path, help = 'Name of the plot')
    parser.add_argument('--metrics_file_name', type = str, help = 'Name of the metrics file you are saving. We recommend classifier name and tested data.')
    parser.add_argument('--plot_name', type = str, help = 'Name of the plot. We recommend classifier name and tested data.')
    args = parser.parse_args()

    # if not os.path.isfile(args.model_path):
    #     print(f'Error: {args.model_path} does not exist (or is no file or not accessible)')
    #     quit()

    if not os.path.isdir(args.dataset_path):
        print(f'Error: {args.dataset_path} does not exist (or is no folder or not accessible)')
        quit()

    aurocs = []
    sample_sizes = []

    metrics_file_path = args.metrics_path / f"{args.metrics_file_name}.txt"
    os.makedirs(args.metrics_path.parent, exist_ok=True)
    with open(metrics_file_path, 'w') as metrics_file:
        for i in os.listdir(args.model_path):
            print(f'Loading model {i}')
            with open(args.model_path / i, 'rb') as file:
                model = pickle.load(file)

            print(f'Loading dataset {args.dataset_path}')
            dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                         loader=lambda path: numpy.load(path),
                                                         extensions=("npy"),
                                                         transform=torchvision.transforms.Compose([
                                                             numpy.squeeze,
                                                             _EEGPreprocessor(250, 0.5, 40),
                                                             EEGSignalToFeaturesFFT(sampling_rate=250)
                                                         ]))
            #if i.__contains__('DWT'):
            #    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
            #                                            loader = lambda path: numpy.load(path),
            #                                            extensions = ("npy"),
            #                                            transform = torchvision.transforms.Compose([
            #                                                numpy.squeeze,
            #                                                _EEGPreprocessor(250, 0.5, 40),
            #                                                EEGSignalToFeaturesDWT('db4', 'symmetric')
            #                                            ]))
            #elif i.__contains__('Welch'):
            #    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
            #                                            loader = lambda path: numpy.load(path),
            #                                            extensions = ("npy"),
            #                                            transform = torchvision.transforms.Compose([
            #                                                numpy.squeeze,
            #                                                _EEGPreprocessor(250, 0.5, 40),
            #                                                EEGSignalToFeaturesWelch(250)
            #                                            ]))
            #elif i.__contains__('CWT'):
            #    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
            #                                                 loader = lambda path: numpy.load(path),
            #                                                 extensions = ("npy"),
            #                                                 transform = torchvision.transforms.Compose([
            #                                                     numpy.squeeze,
            #                                                     _EEGPreprocessor(250, 0.5, 40),
            #                                                     EEGSignalToFeaturesCWT(wavelet='db4')
            #                                                 ])

            print('')
            print(f'Number of samples in dataset: {len(dataset)}')
            print('Classes:')

            for target in Counter([t for _, t in dataset]).most_common():
                print(f'  {dataset.classes[target[0]]}: {target[1]}')

            print('')
            print('Evaluate classifier perfomance...')
            metrics = evaluate_classifier(model, dataset)

            print('')
            print(f'Accuracy at sample {dataset}: {round(metrics.accuracy, 2)}')
            print(f'TPR at sample {dataset}: {round(metrics.tpr, 2)}')
            print(f'TNR at sample {dataset}: {round(metrics.tnr, 2)}')
            print(f'F1 score at sample {dataset}: {round(metrics.f1_score, 2)}')
            print(f'AUROC at sample {dataset}: {round(metrics.auroc, 2)}')
            print('')

            # Write metrics to file
            metrics_file.write(f'Model: {i}\n')
            metrics_file.write(f'Accuracy: {metrics.accuracy}\n')
            metrics_file.write(f'TPR: {metrics.tpr}\n')
            metrics_file.write(f'TNR: {metrics.tnr}\n')
            metrics_file.write(f'F1 score: {metrics.f1_score}\n')
            metrics_file.write(f'AUROC: {metrics.auroc}\n')
            metrics_file.write('\n')

            aurocs.append(metrics.auroc)

            # Extract the sample size from the filename
            sample_size = int(i.split('_')[-2])
            sample_sizes.append(sample_size)

    print(len(aurocs))

    # Sort by sample size
    sorted_indices = np.argsort(sample_sizes)
    sorted_sample_sizes = np.array(sample_sizes)[sorted_indices]
    sorted_aurocs = np.array(aurocs)[sorted_indices]

    # Plot the AUROCs
    plt.plot(sorted_sample_sizes, sorted_aurocs, marker='o')
    plt.xlabel('Sample Size')
    plt.ylabel('AUROC')
    plt.title('AUROC for Different Sample Sizes')

    # Save the plot with a specific name
    plot_filename = args.plot_path / f"{args.plot_name}.png"
    os.makedirs(plot_filename.parent, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(plot_filename)



