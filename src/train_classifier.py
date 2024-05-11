import argparse
from eeg_preprocessing import EEGPreprocessor
import numpy
import os
import pathlib
import pickle
import pywt
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
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


def train_classifier(model, dataset):
     feature_list, target_list = zip(*dataset)
     model.fit(feature_list, target_list)
     return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train a classifier model based on the provided dataset.')
    parser.add_argument('dataset_path', type = pathlib.Path, help = 'Path to the dataset')
    parser.add_argument('model_dump_path', type = pathlib.Path, help = 'Path to file where the trained model is serialized to')

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
         print(f"Error: {args.dataset_path} does not exist (or is no folder or not accessible)")
         quit()

    if os.path.exists(args.model_dump_path):
         print(f"Error: {args.model_dump_path} already exists")
         quit()

    model = GradientBoostingClassifier(n_estimators = 500)

    print('Loading dataset...')
    dataset = torchvision.datasets.DatasetFolder(args.dataset_path,
                                                 loader = lambda path: numpy.load(path),
                                                 extensions = ("npy"),
                                                 transform = torchvision.transforms.Compose([
                                                     numpy.squeeze,
                                                     _EEGPreprocessor(250, 0.5, 60),
                                                     EEGSignalToFeaturesDWT('db4', 'symmetric')
                                                 ]))
    
    print('Training classifier...')
    model = train_classifier(model, dataset)

    print(f'Store classifier as {args.model_dump_path}...')
    os.makedirs(os.path.split(args.model_dump_path)[0], exist_ok = True)
    with open(args.model_dump_path, 'wb') as file:
         pickle.dump(model, file)

    print("Done")
