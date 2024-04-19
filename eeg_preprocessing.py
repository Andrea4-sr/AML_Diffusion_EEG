# Code to preprocess epilepsy and healthy EEG signals

import mne, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import csv

class EEGAnalyzer:

    def __init__(self, data):
        self.data = data  # Assuming data is a list of EEG signals

    def extract_eeg_channels(self, csv_file):
        eeg_channels = set()
        with open(csv_file, 'r') as file:
            #print(file)
            reader = csv.reader(file)
            # Skip headers until reach the channels
            for row in reader:
                #print(row[0].strip())
                if not row[0].startswith('#'):  # Skip lines starting with '#'
                    break

            num_channels = 0
            for row in reader:
                eeg_channel = row[0].strip()
                #print(eeg_channel)
                eeg_channels.add(eeg_channel)
                #print(eeg_channels)
                num_channels += 1

            # Exclude files with less than 6 channels
            if num_channels < 6:
                eeg_channels.clear()
        return eeg_channels

    def find_shared_eeg_channels(self, csv_files):
        common_channels = None
        for file in csv_files:
            channels = self.extract_eeg_channels(file)
            print(common_channels)
            print(len(channels))

            if common_channels is None:
                common_channels = channels
                #print(common_channels)
            elif len(channels) >= 18:
                common_channels = common_channels.intersection(channels)
        return common_channels

class EEGPreprocessor:

    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate

    def bandpass_fitler(self, data, lowcut, highcut):

        nyq = 0.5*self.sampling_rate
        low = lowcut/nyq
        high = highcut/nyq
        order = 2
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    def downsampling(self, data, samples=250):
        return
    def remove_artifacts(self, data, threshold):

        artifacts_epochs = np.any(np.abs(data) > threshold, axis=1)
        cleaned_data = data[~artifacts_epochs]
        return cleaned_data

    def preprocess(self, data, lowcut=0.5, highcut=50, artifact_threshold=100):

        data_filtered = self.bandpass_fitler(data, lowcut, highcut)
        cleaned_data = self.remove_artifacts(data_filtered, artifact_threshold)

        return cleaned_data

    def preprocess_dir(self, dir):
        files = [f for f in os.listdir(dir) if f.endswith('.edf')]
        for file in files:
            raw = mne.io.read_raw_edf(os.path.join(dir, file))
            data = raw.get_data()
            cleaned_data = self.preprocess(data)


if __name__ == "__main__":

    import os

    def get_csv_files(directory):
        csv_files = []
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                csv_files.append(os.path.join(directory, file))
        return csv_files

    # Example usage
    dir =  '/Users/andreasantos/Library/Mobile Documents/com~apple~CloudDocs/Documents/Andrea/Studies/Universities/UZH/Semester 8/Advanced ML/Project/data/edf'
    csv_files = get_csv_files(dir)
    analyzer = EEGAnalyzer(data=csv_files)  # Create an instance of EEGAnalyzer

    # Find shared EEG channels across all CSV files
    shared_channels = analyzer.find_shared_eeg_channels(csv_files)

    print("Shared EEG channels across all files:", shared_channels)
