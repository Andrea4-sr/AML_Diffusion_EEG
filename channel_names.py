import numpy as np
import pyedflib
import os
import tqdm
from scipy.stats import mode

path = 'data/testing/'

channels_of_interest = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]

for i in tqdm.tqdm([x for x in os.listdir(path) if x.endswith('.edf')]):
    f = pyedflib.EdfReader(path + i)
    raw_ = np.empty((0, f.getNSamples()[0]))

    j = [x for x in f.getSignalLabels() if x.__contains__('F3') or x.__contains__('FZ') or x.__contains__('F4') or x.__contains__('C3') or x.__contains__('CZ') or x.__contains__('C4') or x.__contains__('P3') or x.__contains__('PZ') or x.__contains__('P4')]
    j = j[0:9]
    assert j == ['EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'] or \
                ['EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

    print(j)
    # Find the places in the original .getSignalLabels() function where the channels_of_interest are
    k = [f.getSignalLabels().index(x) for x in j]
    print(k)
    f.close()


def data_summary(directory):

    mean_sf_values = []
    mode_sf_values = []
    duration_values = []
    common_channels = []
    channel_set = set()

    for i in [x for x in os.listdir(directory) if x.endswith('.edf')]:
        file_path = os.path.join(directory, i)
        edf_file = pyedflib.EdfReader(file_path)

        # Mean sampling frequency
        signal_freq = edf_file.getSampleFrequency(0)
        mean_sf_values.append(signal_freq)

        # Mode sampling frequency (same as mean since there is only one frequency)
        mode_sf_values.append(signal_freq)

        # Duration in seconds
        duration = edf_file.getFileDuration()
        duration_values.append(duration)

        # Common channels
        channels = edf_file.getSignalLabels()
        common_channels.append(set(channels))

        # Set of channels that are in all files
        if not channel_set:
            channel_set.update(channels)
        else:
            channel_set.intersection_update(channels)

        edf_file.close()

    summary_stats = {
        'mean_sf': np.mean(mean_sf_values),
        'mode_sf': mode(np.asarray(mode_sf_values, dtype=object), keepdims=False),
        'min_duration (s)': np.min(duration_values),
        'max_duration (s)': np.max(duration_values),
        'mean_duration (s)': np.mean(duration_values),
        'std_duration': np.std(duration_values),
        'common_channels': sorted(list(channel_set))
    }

    return summary_stats

print(data_summary('data/testing/'))