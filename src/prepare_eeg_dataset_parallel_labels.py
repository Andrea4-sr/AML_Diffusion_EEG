import multiprocessing
from collections.abc import Callable
from scipy.signal import butter, filtfilt, iirnotch
from typing import Callable
import glob
import numpy
import numpy.typing
import os
import pandas
import pathlib
import pyedflib
import scipy
import tqdm
import random
random.seed(20)

# Import all the helper functions from the original file
from prepare_eeg_dataset import (_filenames_of_edf_csv_pairs, _has_interesting_channel, _merge_ranges,
                                 _inverted_ranges, _eroded_ranges, EdfReader, _get_ranges_for_labels,
                                 _get_ranges_unlabeled, _normalize_channel_name, _resample_signal,
                                 _butter_bandpass, _bandpass_filter, _notch_filter)

def get_ranges_for_specific_labels(csv_file_path: str, channels: 'list[str]', signal_length: float):
    """Returns ranges for specific labels and background."""
    seizure_labels = ["fnsz", "gnsz", "tcsz", "cpsz"]
    seizure_ranges = _get_ranges_for_labels(csv_file_path, seizure_labels, channels, signal_length)
    
    # Get background ranges
    all_seizure_ranges = []
    for label in seizure_ranges:
        all_seizure_ranges.extend(seizure_ranges[label])
    all_seizure_ranges = _merge_ranges(all_seizure_ranges)
    background_ranges = _inverted_ranges(all_seizure_ranges, 0, signal_length)
    
    seizure_ranges["bckg"] = background_ranges
    return seizure_ranges

def process_file(args):
    csv_file_path, edf_file_path, channels_to_extract, target_frequency, signal_length_in_seconds, \
    do_with_samples, erode_ranges_by_secs, notch_channels_by_secs, \
    erode_filter, lowcut, highcut, output_path, max_samples_per_file = args

    num_samples = target_frequency * signal_length_in_seconds
    files_created = 0

    try:
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
                data_per_channel = [_bandpass_filter(signal, lowcut, highcut, target_frequency) for signal in data_per_channel]
                data_per_channel = [_notch_filter(signal, erode_filter, target_frequency) for signal in data_per_channel] if erode_filter is not None else data_per_channel

                min_signal_length = min([len(data) for data in data_per_channel])
                data_per_channel = [data[:min_signal_length] for data in data_per_channel]

                if notch_channels_by_secs is not None:
                    erode_by_samples = notch_channels_by_secs * target_frequency
                    data_per_channel = [data[erode_by_samples:-erode_by_samples] for data in data_per_channel]
                    min_signal_length = min([len(data) for data in data_per_channel])
                
                data_per_channel = numpy.vstack(data_per_channel)

                if min_signal_length >= num_samples:
                    ranges = get_ranges_for_specific_labels(csv_file_path, channels_to_extract, min_signal_length / target_frequency)

                    if erode_ranges_by_secs is not None:
                        ranges = {label: _eroded_ranges(ranges[label], erode_ranges_by_secs) for label in ranges}

                    samples_from_this_file = 0
                    for label in ranges:
                        for r in ranges[label]:
                            start_index = int(r[0] * target_frequency)
                            end_index = int(r[1] * target_frequency)

                            if start_index < 0 or end_index >= min_signal_length:
                                print(f'Warning: {os.path.basename(csv_file_path)}, label {label}: Range [{r[0]} {r[1]}] ({start_index} {end_index}) is outside of signal range (0 {min_signal_length})')

                            for i in range(start_index, end_index, num_samples):
                                if i + num_samples <= end_index and i + num_samples < min_signal_length and samples_from_this_file < max_samples_per_file:
                                    data_to_write = data_per_channel[:, i:i + num_samples]
                                    assert data_to_write.shape == (len(channels_to_extract), num_samples)

                                    files_created += do_with_samples(output_path, data_to_write, os.path.splitext(os.path.basename(edf_file_path))[0], label, i, i + num_samples)
                                    samples_from_this_file += 1
                                if samples_from_this_file >= max_samples_per_file:
                                    break
                            if samples_from_this_file >= max_samples_per_file:
                                break
                        if samples_from_this_file >= max_samples_per_file:
                            break

    except Exception as e:
        print(f"Error processing file {edf_file_path}: {str(e)}")

    return files_created

def prepare_eeg_dataset_parallel(input_path: pathlib.Path,
                                 channels_to_extract: 'list[str]',
                                 target_frequency: int,
                                 signal_length_in_seconds: int,
                                 do_with_samples: Callable[[str, numpy.typing.NDArray, str, str, int, int], int],
                                 erode_ranges_by_secs: None,
                                 notch_channels_by_secs: None,
                                 erode_filter: int,
                                 lowcut: float,
                                 highcut: float,
                                 max_files: None,
                                 num_processes: int,
                                 output_path: str,
                                 max_samples_per_file: int):
    """
    Prepares the TUH EEG dataset at input_path and calls the callable do_with_samples on each prepared sample.
    Samples are saved in subfolders according to their labels.
    This version uses multiprocessing to parallelize the processing.
    """

    filenames_with_csv_and_edf = _filenames_of_edf_csv_pairs(input_path)
    csv_files = [os.path.join(input_path, x + ".csv") for x in filenames_with_csv_and_edf]
    edf_files = [os.path.join(input_path, x + ".edf") for x in filenames_with_csv_and_edf]

    paired_files = list(zip(csv_files, edf_files))
    random.shuffle(paired_files)

    if max_files is not None:
        paired_files = paired_files[:max_files]

    args_list = [(csv_file, edf_file, channels_to_extract, target_frequency, signal_length_in_seconds,
                  do_with_samples, erode_ranges_by_secs, notch_channels_by_secs,
                  erode_filter, lowcut, highcut, output_path, max_samples_per_file) for csv_file, edf_file in paired_files]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(process_file, args_list), total=len(args_list), unit=" files"))

    total_files_created = sum(results)
    print(f"Total files created: {total_files_created}")

def _dump_sample_with_label(output_path: str, signal_data: numpy.typing.NDArray, source_filename: str, label: str, start_index: int, end_index: int) -> int:
    """Dumps signal to .npy file in a subfolder according to its label."""
    os.makedirs(os.path.join(output_path, label), exist_ok=True)
    output_filename = source_filename + "_" + str(start_index).rjust(7, '0') + "_" + str(end_index - start_index) + ".npy"
    output_file_path = os.path.join(output_path, label, output_filename)
    if os.path.exists(output_file_path): return 0
    numpy.save(output_file_path, signal_data)
    return 1

def process_tuh_seizure_dataset_parallel():
    channels_to_extract = ["F3", "C3", "P3", "F4", "C4", "P4"]
    target_frequency = 250
    signal_length_in_seconds = 20
    notch_filter = 60
    lowcut = 0.5
    highcut = 124.9
    num_processes = multiprocessing.cpu_count() - 2
    max_samples_per_file = 20

    input_paths = ["E:/eeg_tuh_seizure_data/train"]
    output_path = "E:/pre_project_data"

    os.makedirs(output_path, exist_ok=True)

    for input_path in input_paths:
        print(f"Processing data from: {input_path}")
        prepare_eeg_dataset_parallel(
            input_path=pathlib.Path(input_path),
            channels_to_extract=channels_to_extract,
            target_frequency=target_frequency,
            signal_length_in_seconds=signal_length_in_seconds,
            do_with_samples=_dump_sample_with_label,
            erode_ranges_by_secs=60,
            notch_channels_by_secs=20,
            erode_filter=notch_filter,
            lowcut=lowcut,
            highcut=highcut,
            max_files=None,
            num_processes=num_processes,
            output_path=output_path + f'/clf',
            max_samples_per_file=max_samples_per_file
        )

    print("Processing complete. All data saved to:", output_path)

if __name__ == "__main__":
    process_tuh_seizure_dataset_parallel()