# Code to preprocess epilepsy and healthy EEG signals

import mne, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import csv
import pyedflib
import os
import tqdm
from scipy.stats import mode

class EEGAnalyzer:

    def __init__(self, data:str, channels:list, repo_dir:str):
        self.data = data  # Assuming data is a path of EEG signals (.edf)
        self.channels = channels
        self.repo_dir = repo_dir

    # def common_channels(self, data):
    #
    #     for i in tqdm.tqdm([x for x in os.listdir(data) if x.endswith('.csv')]):
    #         f = pyedflib.EdfReader(data + i)
    #         raw_ = np.empty((0, f.getNSamples()[0]))
    #
    #         j = [x for x in f.getSignalLabels() if
    #               x.__contains__('C3') or x.__contains__('CZ') or x.__contains__('C4')]
    #         j = j[0:9]
    #         assert j == ['EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE'] or \
    #                ['EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF']
    #
    #         #print(j)
    #         # Find the places in the original .getSignalLabels() function where the channels_of_interest are
    #         k = [f.getSignalLabels().index(x) for x in j]
    #         #print(k)
    #         f.close()

    def select_channels(self,file, central_channels:list):
        return central_channels.intersection(file)

    def proces_csv_files(self, files_dir:str, central_channels:list):

        central_electrode_files = []

        for file in os.listdir(files_dir) if file.endswith(".csv"):
            file_path = os.path.join(files_dir,file)
            eeg_channels = extract_eeg_channels(file_path)



    # def data_summary(self, data, channels, repo_dir):
    #
    #     mean_sf_values = []
    #     mode_sf_values = []
    #     duration_values = []
    #     common_channels = []
    #     channel_set = set()
    #
    #     for i in [x for x in os.listdir(data) if x.endswith('.edf')]:
    #         file_path = os.path.join(data, i)
    #         edf_file = pyedflib.EdfReader(file_path)
    #
    #         # Mean sampling frequency
    #         signal_freq = edf_file.getSampleFrequency(0)
    #         mean_sf_values.append(signal_freq)
    #
    #         # Mode sampling frequency (same as mean since there is only one frequency)
    #         mode_sf_values.append(signal_freq)
    #
    #         # Duration in seconds
    #         duration = edf_file.getFileDuration()
    #         duration_values.append(duration)
    #
    #         # Common channels
    #         channels = edf_file.getSignalLabels()
    #         common_channels.append(set(channels))
    #
    #         # Set of channels that are in all files
    #         if not channel_set:
    #             channel_set.update(channels)
    #         else:
    #             channel_set.intersection_update(channels)
    #
    #         edf_file.close()
    #
    #     summary_stats = {
    #         'mean_sf': np.mean(mean_sf_values),
    #         'mode_sf': mode(np.unique(mode_sf_values, return_counts=True), keepdims=False),
    #         'min_duration (s)': np.min(duration_values),
    #         'max_duration (s)': np.max(duration_values),
    #         'mean_duration (s)': np.mean(duration_values),
    #         'std_duration': np.std(duration_values),
    #         'common_channels': sorted(list(channel_set))
    #     }
    #
    #     # Write summary stats to file
    #     stats_file = os.path.join(repo_dir, 'summary_stats.txt')
    #     with open(stats_file, 'w') as f:
    #         for key, value in summary_stats.items():
    #             f.write(f"{key}: {value}\n")
    #
    #     return summary_stats


    # dist plots of amount of seizure activity in files - per channel
    # one label per file (multiple seizures per file)
    # method to discard files with no central electrodes
    #

    def extract_eeg_channels(self, edf_file):
        eeg_channels = set()
        with open(edf_file, 'r') as file:
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

        #     # Exclude files with less than 6 channels
        #     if num_channels < 6:
        #         eeg_channels.clear()
        # return eeg_channels

    # def find_shared_eeg_channels(self, edf_files):
    #     common_channels = None
    #     for file in edf_files:
    #         channels = self.extract_eeg_channels(file)
    #         print(common_channels)
    #         print(len(channels))
    #
    #         if common_channels is None:
    #             common_channels = channels
    #             #print(common_channels)
    #         elif len(channels) >= 6:
    #             common_channels = common_channels.intersection(channels)
    #     return common_channels

    def find_noisy_edfs(self, data, repo_dir, threshold_multiplier):

        noisy_edfs = []

        for file in tqdm.tqdm([f for f in os.listdir(data) if f.endswith('.edf')]):
            file_path = os.path.join(data, file)
            try:
                edf_file = pyedflib.EdfReader(file_path)
                num_channels = edf_file.signals_in_file
                #print(num_channels)
                all_eeg_data = np.zeros((num_channels, edf_file.getNSamples()[0]))
                #print(edf_file.getNSamples()[0])
                for i in range(num_channels):
                    #print(i)
                    all_eeg_data[i, :] = edf_file.readSignal(i) # extract signals and iterate over these
                print(all_eeg_data.shape)
                std = np.std(all_eeg_data)
                edf_file.close()

                noise_threshold = threshold_multiplier*std
                if noise_threshold > 0:
                    noisy_edfs.append((file_path, noise_threshold))
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if noisy_edfs:
            with open(os.path.join(repo_dir, 'noisy_edfs.txt'), 'w') as f:
                for noisy_file, threshold in noisy_files:
                    f.write(f"Noisy file: {os.path.basename(noisy_file)}, Threshold: {threshold}\n")
                print("Noisy files and their thresholds written to 'noisy_files_thresholds.txt'.")

        else:
            print("No noisy EDFs found")



class EEGPreprocessor:

    def __init__(self, data): #resample to 250Hz
        self.data = data

    def bandpass_fitler(self, data, lowcut, highcut):

        nyq = 0.5*self.sampling_rate
        low = lowcut/nyq
        high = highcut/nyq
        order = 2
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    def resampling(self, data, samples=250):
        resampled_raw = data.resample(samples)
        resampled_data, _ = resampled_raw[:, :]
        return resampled_data
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

    # Example usage
    dir =  '/Users/andreasantos/Library/Mobile Documents/com~apple~CloudDocs/Documents/Andrea/Studies/Universities/UZH/Semester 8/Advanced ML/Project/data/edf/'
    repo_dir = '/Users/andreasantos/Projects/Coding/AML_EEG/data/'
    interesting_channels = ["C3", "Cz", "C4"]
    analyzer = EEGAnalyzer(data=dir, channels=interesting_channels, repo_dir=repo_dir)  # Create an instance of EEGAnalyzer

    # Find shared EEG channels across all CSV files
    shared_channels = analyzer.common_channels(dir)
    stats = analyzer.data_summary(dir, interesting_channels, repo_dir=repo_dir)

    noisy_edfs = analyzer.find_noisy_edfs(dir, repo_dir, threshold_multiplier=0.5)
