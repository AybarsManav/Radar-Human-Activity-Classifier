import math
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import butter, lfilter, spectrogram

bin_indl = 10
bin_indu = 30

# Helper functions

def get_spectrogram_parameters(T_sweep):
    MD = {}
    MD['PRF'] = 1 / T_sweep
    MD['TimeWindowLength'] = 200
    MD['OverlapFactor'] = 0.95
    MD['OverlapLength'] = round(MD['TimeWindowLength'] * MD['OverlapFactor'])
    MD['Pad_Factor'] = 4
    MD['FFTPoints'] = MD['Pad_Factor'] * MD['TimeWindowLength']

    return MD

def get_mti_plot_params(T_sweep, Data_range_MTI):
    MD = get_spectrogram_parameters(T_sweep)
    MD['DopplerBin'] = MD['PRF'] / MD['FFTPoints']
    MD['DopplerAxis'] = np.arange(-MD['PRF'] / 2, MD['PRF'] / 2, MD['DopplerBin'])
    MD['WholeDuration'] = Data_range_MTI.shape[1] / MD['PRF']
    MD['NumSegments'] = int(
        np.floor((Data_range_MTI.shape[1] - MD['TimeWindowLength']) /
                 np.floor(MD['TimeWindowLength'] * (1 - MD['OverlapFactor']))))

    return MD

def odd_number(x):
    y = int(np.floor(x))
    if y % 2 == 0:
        y = int(np.ceil(x))
    if y % 2 == 0:
        y += 1
    return y

def get_dataset_batches(dataset_dir):
    return list(sorted(filter(lambda d: d.endswith('Dataset'), os.listdir(dataset_dir))))

def get_dataset_files(dataset_dir, dataset_batch):
    return sorted(os.listdir(str(os.path.join(dataset_dir, dataset_batch))))

def read_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = f.read()
        split_data = raw_data.split('\n')

        f_c = float(split_data[0])                          # Center frequency
        T_sweep = float(split_data[1]) / 1000               # Sweep time in seconds
        NTS = int(split_data[2])                            # Number of time samples per sweep
        bw = float(split_data[3])                           # FMCW Bandwidth
        data = split_data[4:]                               # Raw data in I + j * Q format

        fs = NTS/T_sweep                                    # Sampling frequency (ADC)
        record_length = float(len(data)) / fs               # Length of recording in seconds
        n_chirps = math.floor(record_length / T_sweep)      # Number of chirps


        complex_data = np.array([complex(d.replace('i', 'j')) for d in data if len(d.strip()) > 0])

        return f_c, T_sweep, NTS, bw, fs, n_chirps, complex_data


def data_range(complex_data, n_chirps, NTS):
    Data_time = np.reshape(complex_data, (n_chirps, NTS)).T
    win = np.ones_like(Data_time)

    tmp = np.fft.fftshift(np.fft.fft(Data_time * win, axis=0), axes=0)

    Data_range = tmp[NTS // 2:NTS, :]

    return Data_range


def mti_filter(Data_range):
    ns = odd_number(Data_range.shape[1]) - 1

    Data_range_MTI = np.zeros((Data_range.shape[0], ns), dtype=np.complex128)

    b, a, *_ = butter(4, 0.0075, btype='high', analog=False)

    for k in range(Data_range.shape[0]):
        Data_range_MTI[k, :ns] = lfilter(b, a, Data_range[k, :ns])

    return Data_range_MTI





def get_spectrogram(Data_range, T_sweep):
    MD = get_spectrogram_parameters(T_sweep)
    Data_spec = 0

    for RBin in range(bin_indl, bin_indu + 1):
        _, _, s = spectrogram(
            Data_range[RBin, :],
            nperseg=MD['TimeWindowLength'],
            noverlap=MD['OverlapLength'],
            nfft=MD['FFTPoints'],
            fs=MD['PRF'],
            return_onesided=False,
            mode='complex'
        )
        Data_spec += np.abs(np.fft.fftshift(s, axes=0))

    return Data_spec


def get_spectrogram_MTI(Data_range_MTI, T_sweep):
    MD = get_spectrogram_parameters(T_sweep)

    Data_spec_MTI = 0

    for RBin in range(bin_indl, bin_indu + 1):
        _, _, s = spectrogram(
            Data_range_MTI[RBin, :],
            nperseg=MD['TimeWindowLength'],
            noverlap=MD['OverlapLength'],
            nfft=MD['FFTPoints'],
            fs=MD['PRF'],
            window='hamming',
            return_onesided=False,
            mode='complex'

        )
        Data_spec_MTI += np.abs(np.fft.fftshift(s, axes=0))

    return np.flipud(Data_spec_MTI)

def plot_range_MTI(Data_range_MTI):
    magnitude_db = 20 * np.log10(np.abs(Data_range_MTI) + 1e-12)

    # Plot
    plt.figure()
    plt.imshow(magnitude_db, aspect='auto', cmap='jet', origin='lower')

    # Labels and title
    plt.xlabel('No. of Sweeps')
    plt.ylabel('Range bins')
    plt.title('Range Profiles after MTI filter')

    # Y-axis limit
    plt.ylim(1, Data_range_MTI.shape[0] - 1)

    # Match MATLAB clim logic
    vmax = np.max(magnitude_db)
    print(np.argmax(magnitude_db))
    vmin = vmax - 60
    plt.clim(vmin, vmax)

    plt.colorbar(label='Magnitude (dB)')
    plt.draw()
    plt.show()

def plot_spec_MTI(T_sweep, Data_spec_MTI):
    MD = get_mti_plot_params(T_sweep, Data_spec_MTI)
    velocity_axis = MD['DopplerAxis'] * 3e8 / 2 / 5.8e9
    MD['TimeAxis'] = np.linspace(0, MD['WholeDuration'], Data_spec_MTI.shape[1])

    # Plot
    plt.figure()
    plt.imshow(
        20 * np.log10(Data_spec_MTI + 1e-12),
        extent=(MD['TimeAxis'][0], MD['TimeAxis'][-1], velocity_axis[0], velocity_axis[-1]),
        aspect='auto',
        cmap='jet',
        origin='lower'
    )
    plt.colorbar(label='Magnitude (dB)')
    plt.ylim([-6, 6])
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Velocity [m/s]', fontsize=16)
    plt.title('Spectrogram after MTI filter')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # CLim equivalent
    vmax = np.max(20 * np.log10(Data_spec_MTI + 1e-12))
    plt.clim(vmax - 40, vmax)

    plt.show()

# End of helper functions

def preprocess_file(file_path, plot_range_mti = False, plot_spec_mti = False):
    f_c, T_sweep, NTS, bw, fs, n_chirps, complex_data = read_data(file_path)
    Data_range = data_range(complex_data, n_chirps, NTS)
    Data_range_MTI = mti_filter(Data_range)

    # Remove first range bin
    Data_range_MTI = Data_range_MTI[1:Data_range_MTI.shape[0], :]
    Data_range = Data_range[1:Data_range_MTI.shape[0], :]

    if plot_range_mti:
        plot_range_MTI(Data_range_MTI)

    Data_spec = get_spectrogram(Data_range, T_sweep)
    Data_spec_MTI = get_spectrogram_MTI(Data_range, T_sweep)

    if plot_spec_mti:
        plot_spec_MTI(T_sweep, Data_spec_MTI)

    return Data_spec, Data_spec_MTI


def get_labels(file_name):
    """
    Extract labels from filenames in the format 'PxxAxxRxx.dat'.

    Args:
        file_name (str): The name of the file.
            Each filename must follow the format 'PxxAxxRxx.dat', where:
            - 'Pxx' indicates the person ID (e.g., P01 to P72),
            - 'Axx' indicates the activity ID (e.g., A01 to A06),
            - 'Rxx' indicates the repetition number (e.g., R01 to R03).

    Returns:
        tuple:
            - number1 (list of int): Person IDs (1 to 72).
            - number2 (list of int): Activity IDs:
                1 - walking,
                2 - sitting,
                3 - standing,
                4 - drink water,
                5 - pick,
                6 - fall.
            - number3 (list of int): Repetition counts (1 to 3).

    Description:
        This function parses each filename and extracts three numeric labels:
        the performer of the activity (person), the type of activity, and the
        repetition number. It supports both individual filenames and lists of filenames.
    """

    num1, num2, num3 = '', '', ''
    i = 0

    # Skip until 'P'
    while i < len(file_name) and file_name[i] != 'P':
        i += 1
    i += 1  # move past 'P'

    # Collect person ID until 'A'
    while i < len(file_name) and file_name[i] != 'A':
        num1 += file_name[i]
        i += 1
    i += 1  # move past 'A'

    # Collect activity ID until 'R'
    while i < len(file_name) and file_name[i] != 'R':
        num2 += file_name[i]
        i += 1
    i += 1  # move past 'R'

    # Collect repetition ID until '.' or end
    while i < len(file_name) and file_name[i] not in ('.', 'D', 'R'):
        num3 += file_name[i]
        i += 1

    return int(num1), int(num2), int(num3)

if __name__ == '__main__':
    dataset_dir = 'datasets'
    datasets_batches = get_dataset_batches(dataset_dir)

    spectrograms = []
    spectrograms_MTI = []
    labels = []

    for batch in datasets_batches:
        files = get_dataset_files(dataset_dir, batch)
        for i, file in enumerate(files):
            if 'Copy' in file or '(' in file:
                print(f'{batch} | [{i + 1}/{len(files)}]: {file} ignored for unrecognized file name.')
                continue

            t = time.time()
            dataset_file_path = os.path.join(dataset_dir, batch, file)
            person, activity, repetition = get_labels(file)
            Data_spec, Data_spec_MTI = preprocess_file(dataset_file_path, plot_range_mti=False, plot_spec_mti=False)

            spectrograms.append(Data_spec)
            spectrograms_MTI.append(Data_spec_MTI)
            # Only interested in activity
            labels.append(activity)

            print(f'{batch} | [{i + 1}/{len(files)}]: {file} in {time.time() - t:.2f} seconds. Activity: {activity}.')

    os.makedirs('preprocessed_data', exist_ok=True)
    with open(os.path.join('preprocessed_data', 'spectrograms.pkl'), 'wb') as f:
        pickle.dump((spectrograms, spectrograms_MTI, np.array(labels)), f)
