import argparse
from collections.abc import Callable
import glob
import os
import pandas
import pathlib
import pickle
import pyedflib
import scipy
import tqdm


def _filenames_of_edf_csv_pairs(edf_dir: str):
    edf_files = glob.glob(os.path.join(edf_dir, "*.edf"))
    filenames = [os.path.splitext(os.path.basename(x))[0] for x in edf_files]
    filenames_with_csv_and_edf = [x for x in filenames if os.path.isfile(os.path.join(edf_dir, x + ".csv")) ]# and os.path.isfile(os.path.join(edf_dir, ".edf"))] # not necessary since the initial list is based on edf files existing.
    return sorted(filenames_with_csv_and_edf)


def _has_interesting_channel(channel_spec: str, channels: list[str]):
    channels_in_spec = channel_spec.split('-')
    for channel in channels_in_spec:
        if channel in channels:
            return True
    return False


def _merge_ranges(ranges: list[tuple[int, int]]):
    for range in ranges:
        assert range[0] < range[1]
        if range[0] >= range[1]:
            raise ValueError()
        
    sorted_ranges = sorted(ranges, key = lambda x: x[0])
    i = 0

    while i < len(sorted_ranges) - 1:
        if sorted_ranges[i][1] >= sorted_ranges[i + 1][0]:
            sorted_ranges[i] = (sorted_ranges[i][0], max(sorted_ranges[i][1], sorted_ranges[i + 1][1]))
            del sorted_ranges[i + 1]
        else:
            i += 1
    
    return sorted_ranges


def _inverted_ranges(ranges: list[tuple[int, int]], begin: float, end: float):
    if len(ranges) == 0:
        return [(begin, end)]
    
    inverted = []

    if begin < ranges[0][0]:
        inverted.append((begin, ranges[0][0]))

    for i in range(len(ranges) - 1):
        if ranges[i][1] < ranges[i + 1][0]:
            inverted.append((ranges[i][1], ranges[i + 1][0]))
    
    if end > ranges[-1][1]:
        inverted.append((ranges[-1][1], end))
    
    return inverted


def _eroded_ranges(ranges: list[tuple[int, int]], erode_by: float):
    i = 0

    while i < len(ranges):
        ranges[i] = (ranges[i][0] + erode_by, ranges[i][1] - erode_by)
        if ranges[i][0] >= ranges[i][1]:
            del ranges[i]
        else:
            i += 1
    
    return ranges


class EdfReader(object):
    def __init__(self, path: str):
        self.path = path
    
    def __enter__(self):
        self.file = pyedflib.EdfReader(self.path)
        return self.file
    
    def __exit__(self, *args):
        self.file.close()


def _get_ranges_for_labels(csv_file_path: str, labels: list[str], channels: list[str], _unused_signal_length: float):
    csv_data = pandas.read_csv(csv_file_path, delimiter = ",", skiprows = 5)
    interesting_data = csv_data[csv_data['label'].isin(labels) &
                                csv_data['channel'].apply(lambda x: _has_interesting_channel(x, channels))]
    
    ranges_for_labels = {}

    for label in labels:
        data_for_label = interesting_data[interesting_data['label'] == label]

        if not data_for_label.empty:
            ranges = list(zip(data_for_label['start_time'], data_for_label['stop_time']))
            try:
                merged_ranges = _merge_ranges(ranges)
                ranges_for_labels[label] = merged_ranges
            except ValueError:
                print(f"{csv_file_path}: Could not determine ranges for label {label}: {ranges}")
                pass
    
    return ranges_for_labels


def _get_ranges_unlabeled(csv_file_path: str, _unused_labels: list[str], channels: list[str], signal_length: float):
    csv_data = pandas.read_csv(csv_file_path, delimiter = ",", skiprows = 5)
    interesting_data = csv_data[csv_data['channel'].apply(lambda x: _has_interesting_channel(x, channels))]

    ranges = []

    if not interesting_data.empty:
        ranges = list(zip(interesting_data['start_time'], interesting_data['stop_time']))

        try:
            ranges = _merge_ranges(ranges)
            ranges = _inverted_ranges(ranges, 0, signal_length)
        except ValueError:
            print(f"{csv_file_path}: Could not determine ranges for labels: {ranges}")
    
    return ranges


def _normalize_channel_name(name: str):
    if name.startswith('EEG '):
        name = name[4:]
    name = name.split('-')[0]
    return name


def _resample_signal(signal, source_frequency: float, target_frequency: float):
    if source_frequency == target_frequency:
        return signal
    else:
        return scipy.signal.resample(signal, int(len(signal) * target_frequency / source_frequency))


def prepare_eeg_dataset(input_path: pathlib.Path,
                        output_path: pathlib.Path,
                        channels_to_extract: list[str],
                        classes_to_extract: list[str],
                        target_frequency: int,
                        signal_length_in_seconds: int,
                        extract_ranges_from_csv: Callable[[str, list[str], list[str], float], dict[str, list[float, float]]],
                        erode_channels_by_secs: float | None = None,
                        erode_ranges_by_secs: float | None = None,
                        max_files: int | None = None):
    filenames_with_csv_and_edf = _filenames_of_edf_csv_pairs(input_path)
    csv_files = [os.path.join(input_path, x + ".csv") for x in filenames_with_csv_and_edf]
    edf_files = [os.path.join(input_path, x + ".edf") for x in filenames_with_csv_and_edf]

    num_samples = target_frequency * signal_length_in_seconds
    files_created = 0

    os.makedirs(output_path, exist_ok = True)
    progress = tqdm.tqdm(total = len(edf_files) if max_files is None else max_files, unit = " files")

    try:
        for csv_file_path, edf_file_path in zip(csv_files, edf_files):
            with EdfReader(edf_file_path) as edf_file:
                channels = [_normalize_channel_name(n) for n in edf_file.getSignalLabels()]
                interesting_channel_indices = []
                skip_file = False

                for interesting_channel in channels_to_extract:
                    try:
                        interesting_channel_indices.append(channels.index(interesting_channel))
                    except:
                        print(f'{os.path.basename(edf_file_path)} does not contain required channel {interesting_channel}. Skipping file.')
                        skip_file = True
                        break
                
                if not skip_file:
                    data_per_channel = [edf_file.readSignal(channel) for channel in interesting_channel_indices]
                    frequency_per_channel = [edf_file.getSampleFrequency(channel) for channel in interesting_channel_indices]
                    data_per_channel = [_resample_signal(signal, source_freq, target_frequency) for signal, source_freq in zip(data_per_channel, frequency_per_channel)]
                    min_signal_length = min([len(data) for data in data_per_channel])
                    data_per_channel = [data[:min_signal_length] for data in data_per_channel]

                    if erode_channels_by_secs is not None:
                        erode_by_samples = erode_channels_by_secs * target_frequency
                        data_per_channel = [data[erode_by_samples:-erode_by_samples] for data in data_per_channel]
                        min_signal_length = min([len(data) for data in data_per_channel])

                    if min_signal_length >= num_samples:
                        ranges = extract_ranges_from_csv(csv_file_path, classes_to_extract, channels_to_extract, min_signal_length / target_frequency)

                        if erode_ranges_by_secs is not None:
                            ranges = {label: _eroded_ranges(ranges[label], erode_ranges_by_secs) for label in ranges}

                        for label in ranges:
                            os.makedirs(os.path.join(output_path, label), exist_ok = True)
                            for r in ranges[label]:
                                start_index = int(r[0] * target_frequency)
                                end_index = int(r[1] * target_frequency)
                                for i in range(start_index, end_index, num_samples):
                                    if end_index - i >= num_samples:
                                        data_to_write = {channels[channel]: data_per_channel[index][i:i + num_samples] for index, channel in enumerate(interesting_channel_indices)}
                                        output_filename = os.path.splitext(os.path.basename(edf_file_path))[0] + "_" + str(i).rjust(7, '0') + "_" + str(num_samples) + ".pkl"
                                        with open(os.path.join(output_path, label, output_filename), 'wb') as output_file:
                                            pickle.dump(data_to_write, output_file, pickle.HIGHEST_PROTOCOL)
                                            files_created += 1
                                            if max_files is not None:
                                                progress.update()
                                                if files_created >= max_files:
                                                    raise StopIteration()
            if max_files is None:
                progress.update()
    except StopIteration:
        pass
    
    progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process EDF files to create files with EEG data according to specification.')
    parser.add_argument('frequency', type = int, help = 'Frequency the extracted data is sampled in')
    parser.add_argument('duration', type = int, help = 'Length of signals in seconds written to output files')
    parser.add_argument('channels', type = str, help = 'List of names of channels to extract (e.g. "F1,F2,C4,P7")')
    parser.add_argument('classes', type = str, help = 'List of classes to extracted (e.g. "bckg,seiz,fnsz")')
    parser.add_argument('input_path', type = pathlib.Path, help = 'Path to folder containing EDF and CSV files')
    parser.add_argument('output_path', type = pathlib.Path, help = 'Path to folder where output files are written')
    parser.add_argument('--max-files', type = int, help = 'Maximum number of files to create')

    args = parser.parse_args()

    channels_to_extract = args.channels.split(",")
    classes_to_extract = args.classes.split(",")

    if args.max_files is not None and args.max_files < 1:
        print("Error: --max_files must be at least 1")
        quit()


    # Extract samples for the requested channels and classes.
    prepare_eeg_dataset(args.input_path,
                        args.output_path,
                        channels_to_extract,
                        classes_to_extract,
                        args.frequency,
                        args.duration,
                        _get_ranges_for_labels,
                        120,
                        None,
                        args.max_files)

    # Extract samples for the requested channels that do not belong to a class
    # and put them in a class "non_seizure".
    prepare_eeg_dataset(args.input_path,
                        args.output_path,
                        channels_to_extract,
                        classes_to_extract,
                        args.frequency,
                        args.duration,
                        lambda csv_file_path, labels, channels, signal_length: {"non_seizure": _get_ranges_unlabeled(csv_file_path, labels, channels, signal_length)},
                        120,
                        20,
                        args.max_files)
