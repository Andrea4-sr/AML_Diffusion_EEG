import argparse
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
from scipy.signal import butter, filtfilt
import torchvision
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
import random
random.seed(20)




class EEGPreprocessor:
    def __init__(self, sampling_rate, lowcut, highcut):
        # sampling rate of the EEG signal
        self.sampling_rate = sampling_rate
        # lower cutoff frequency for the bandpass filter
        self.lowcut = lowcut
        # higher cutoff frequency for the bandpass filter
        self.highcut = highcut
    
    def __call__(self, data):
        return self.bandpass_fitler(data, self.sampling_rate, self.lowcut, self.highcut)
    
    def bandpass_fitler(self, data, sampling_rate, lowcut, highcut):

        # calculate the nyquist frequency -- half of the sampling rate
        nyq = 0.5 * sampling_rate
        # normalise lower cutoff frequency with respect to the nyquist frequency
        low = lowcut/nyq
        # normalise higher cutoff frequency with respect to the nyquist frequency
        high = highcut/nyq


        order = 2

        # Butterworth bandpass filter
        # returns filter coefficients
        b, a = butter(order, [low, high], btype='band')

        # Applies forward-backward filtering to ensure zero phase distortion
        filtered_data = filtfilt(b, a, data)
        return filtered_data

# Signal to feature vector composed of std and percentiles
def _signal_to_features(signal):
     signal = signal.squeeze()
     std = numpy.std(signal)
     percentiles = numpy.percentile(signal, [10, 20, 30, 40, 50, 60, 70, 80, 90])
     return numpy.append(std, percentiles)

# Extract features from signals using Discrete Wavelet Transform (DWT)
class EEGSignalToFeaturesDWT:
     def __init__(self, wavelet, mode):
          self.wavelet = wavelet
          self.mode = mode
     
     def __call__(self, signal):
          features = [_signal_to_features(n) for n in pywt.wavedec(signal, wavelet = self.wavelet, mode = self.mode)]
          return numpy.asarray(features).flatten()

# Extract features from signals using Fast Fourier Transform (FFT)
class EEGSignalToFeaturesFFT:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, signal):
        fft_result = np.fft.fft(signal)
        magnitude_spectrum = numpy.abs(fft_result)
        phase_spectrum = numpy.angle(fft_result)
        features = np.concatenate((magnitude_spectrum, phase_spectrum))
        return features

# Extract features from signals using Welch's method
class EEGSignalToFeaturesWelch:
    def __init__(self, sampling_rate, nperseg=None):
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg or sampling_rate // 2

    def __call__(self, signal):
        frequency, power = welch(signal, fs=self.sampling_rate, nperseg=self.nperseg)
        power = power[:70]
        power = np.log(power)
        return power

# Train a classifier on a given dataset
def train_classifier(model, dataset):
     feature_list, target_list = zip(*dataset)
     model.fit(feature_list, target_list)
     return model

# Combinations of feature extraction methods and classifiers
def feature_classifier_combos(feat_ext_methods, classifiers):

    combos = list(itertools.product(feat_ext_methods, classifiers))
    return combos

# Load and transform dataset
def load_and_transform_data(dataset, transform):
    processed_data = []
    for path, target in dataset.samples:
        data = np.load(path)
        transformed_data = transform(data)
        processed_data.append((transformed_data, target))
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train a classifier model based on the provided dataset.')
    parser.add_argument('--dataset_path', type = pathlib.Path, help = 'Path to the dataset')
    parser.add_argument('--model_dump_path', type = pathlib.Path, help = 'Path to file where the trained model is serialized to')
    parser.add_argument('--n_samples', type = int, default = 1001, help = 'Number of samples to use for training')

    args = parser.parse_args()

    # check if dataset path exists
    if not os.path.isdir(args.dataset_path):
         print(f"Error: {args.dataset_path} does not exist (or is no folder or not accessible)")
         quit()

    # check if model dump path exists
    if os.path.exists(args.model_dump_path):
         print(f"Error: {args.model_dump_path} already exists")
         quit()

    # define models and feature extractors to use
    models = [svm.SVC(probability=True)] #GradientBoostingClassifier(n_estimators = 100), , MLPClassifier()]
    feature_extractors = [EEGSignalToFeaturesFFT] # [EEGSignalToFeaturesWelch, , EEGSignalToFeaturesDWT] #EEGSignalToFeaturesCWT]
    combos = feature_classifier_combos(feature_extractors, models)

    for feature_extractor, classifier in combos:

        model_name = f"{feature_extractor.__name__}_{classifier.__class__.__name__}"
        model_dump_path = args.model_dump_path / f"{model_name}.pkl"
        print(f'Using feature extractor: {feature_extractor.__name__}, and model: {classifier.__class__.__name__}')


        print('Loading dataset...')

        # transform data
        transform = torchvision.transforms.Compose([numpy.squeeze,
                                                   EEGPreprocessor(250, 0.5, 40),
                                                   EEGSignalToFeaturesFFT(sampling_rate=250)
                                                   ])

        dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                     loader=lambda path: np.load(path),
                                                     extensions=("npy",),
                                                     transform=transform)

        # apply transformations to the dataset
        processed_data = load_and_transform_data(dataset, transform)
        # shuffle dataset
        random.shuffle(processed_data)

        # train different classifiers at with increasing number of samples
        for sample_size in range(200, args.n_samples, 200):

            # Ensure the sample dataset contains both classes
            sample_dataset, _ = train_test_split(processed_data, train_size=sample_size,
                                                 stratify=[target for _, target in processed_data])

            class_counts = Counter([target for _, target in sample_dataset])
            if len(class_counts) < 2:
                print(f"Error: Not enough classes in the sample of size {sample_size}.")
                continue

            print(f'Training classifier with {sample_size} samples...')
            model = train_classifier(classifier, sample_dataset)

            model_name = f"{feature_extractor.__name__}_{classifier.__class__.__name__}_{sample_size}_samples.pkl"
            model_dump_path = args.model_dump_path / model_name

            print(f'Storing classifier as {model_dump_path}...')
            os.makedirs(model_dump_path.parent, exist_ok=True)
            with open(model_dump_path, 'wb') as file:
                pickle.dump(model, file)

            print("Done")