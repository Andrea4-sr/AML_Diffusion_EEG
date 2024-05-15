import argparse
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
from scipy.signal import welch
import torchvision
import itertools



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

# class EEGSignalToFeaturesCWT:
#
#     def __init__(self, wavelet):
#         self.wavelet = wavelet
#         self.alpha_range = (8, 12)
#         self.beta_range = (12, 30)
#     def _calculate_medium_scales(self, sampling_rate):
#         alpha_scales = pywt.scale2frequency(self.wavelet, np.arange(64, sampling_rate)) * sampling_rate
#         beta_scales = pywt.scale2frequency(self.wavelet, np.arange(64, sampling_rate)) * sampling_rate
#
#         alpha_medium_scales = alpha_scales[(alpha_scales >= self.alpha_range[0]) & (alpha_scales <= self.alpha_range[1])]
#         beta_medium_scales = beta_scales[(beta_scales >= self.beta_range[0]) & (beta_scales <= self.beta_range[1])]
#
#         medium_scales = np.concatenate((alpha_medium_scales, beta_medium_scales))
#         return medium_scales
#
#     def __call__(self, signal):
#         medium_scales = self._calculate_medium_scales(sampling_rate=250)
#
#         features = []
#         for scale in medium_scales:
#             coefficients, _ = pywt.cwt(signal, scales=[scale], wavelet=self.wavelet)
#             features.append(coefficients.flatten())
#         features_array = np.asarray(features)
#         print(features_array.shape)
#         exit()
#         return np.asarray(features).flatten()

class EEGSignalToFeaturesWelch:
    def __init__(self, sampling_rate, nperseg=None):
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg or sampling_rate // 2

    def __call__(self, signal):
        frequency, power = welch(signal, fs=self.sampling_rate, nperseg=self.nperseg)
        power = power[:70]
        power = np.log(power)
        return power


def train_classifier(model, dataset):
     feature_list, target_list = zip(*dataset)
     model.fit(feature_list, target_list)
     return model

def feature_classifier_combos(feat_ext_methods, classifiers):

    combos = list(itertools.product(feat_ext_methods, classifiers))
    return combos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train a classifier model based on the provided dataset.')
    parser.add_argument('--dataset_path', type = pathlib.Path, help = 'Path to the dataset')
    parser.add_argument('--model_dump_path', type = pathlib.Path, help = 'Path to file where the trained model is serialized to')

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
         print(f"Error: {args.dataset_path} does not exist (or is no folder or not accessible)")
         quit()

    if os.path.exists(args.model_dump_path):
         print(f"Error: {args.model_dump_path} already exists")
         quit()

    models = [GradientBoostingClassifier(n_estimators = 100), svm.SVC(probability=True), MLPClassifier()]
    feature_extractors = [EEGSignalToFeaturesWelch, EEGSignalToFeaturesFFT, EEGSignalToFeaturesDWT] #EEGSignalToFeaturesCWT]
    combos = feature_classifier_combos(feature_extractors, models)

    for feature_extractor, classifier in combos:

        model_name = f"{feature_extractor.__name__}_{classifier.__class__.__name__}"
        model_dump_path = args.model_dump_path / f"{model_name}.pkl"
        print(f'Using feature extractor: {feature_extractor.__name__}, and model: {classifier.__class__.__name__}')

        if feature_extractor == EEGSignalToFeaturesDWT:
            print('Loading dataset...')
            dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                         loader = lambda path: numpy.load(path),
                                                         extensions = ("npy"),
                                                         transform = torchvision.transforms.Compose([
                                                            numpy.squeeze,
                                                            _EEGPreprocessor(250, 0.5, 40),
                                                            EEGSignalToFeaturesDWT('db4', 'symmetric')
                                                         ]))
        #elif feature_extractor == EEGSignalToFeaturesCWT:
        #    print('Loading dataset...')
        #    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
        #                                                 loader = lambda path: numpy.load(path),
        #                                                 extensions = ("npy"),
        #                                                 transform = torchvision.transforms.Compose([
        #                                                    numpy.squeeze,
        #                                                    _EEGPreprocessor(250, 0.5, 60),
        #                                                    EEGSignalToFeaturesCWT(wavelet='db4')
        #                                                ]))
        elif feature_extractor == EEGSignalToFeaturesFFT:
            print('Loading dataset...')
            dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                         loader = lambda path: numpy.load(path),
                                                         extensions = ("npy"),
                                                         transform = torchvision.transforms.Compose([
                                                            numpy.squeeze,
                                                            _EEGPreprocessor(250, 0.5, 40),
                                                            EEGSignalToFeaturesFFT(sampling_rate=250)
                                                        ]))
        elif feature_extractor == EEGSignalToFeaturesWelch:
            print('Loading dataset...')
            dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                         loader = lambda path: numpy.load(path),
                                                         extensions = ("npy"),
                                                         transform = torchvision.transforms.Compose([
                                                            numpy.squeeze,
                                                            _EEGPreprocessor(250, 0.5, 40),
                                                            EEGSignalToFeaturesWelch(sampling_rate=250)
                                                        ]))
        for sample_size in range(200, 1001, 200):
            sample_dataset = dataset.samples[:sample_size]
            print(f'Training classifier with {sample_size} samples...')
            model = train_classifier(classifier, sample_dataset)

            model_name = f"{feature_extractor.__name__}_{classifier.__class__.__name__}_{sample_size}_samples.pkl"
            model_dump_path = args.model_dump_path / model_name

            print(f'Storing classifier as {model_dump_path}...')
            os.makedirs(model_dump_path.parent, exist_ok=True)
            with open(model_dump_path, 'wb') as file:
                pickle.dump(model, file)

            print("Done")

        #print('Training classifier...')
        #model = train_classifier(classifier, dataset)

        #print(f'Storing classifier as {model_dump_path}...')
        #os.makedirs(model_dump_path.parent, exist_ok=True)
        #with open(model_dump_path, 'wb') as file:
        #    pickle.dump(model, file)

        #print("Done")