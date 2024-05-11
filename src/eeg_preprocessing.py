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

    # Automatic sample rejection for small std's (measurement error samples) or very high std's (noise)

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

    def bandpass_fitler(self, data, sampling_rate, lowcut, highcut):

        nyq = 0.5 * sampling_rate
        low = lowcut/nyq # lowcut 0.5Hz
        high = highcut/nyq #highcut 60hz
        order = 2
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    # seizures are visible in range 2-10hz
    # pytorch data folder true




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
