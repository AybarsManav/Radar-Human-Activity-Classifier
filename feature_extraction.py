import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_extraction(spectrogram, f, t):
    """
    Extract features from the input data.
    
    Parameters:
    spectrogram: Radar spectrogram for feature extraction.
    f: Frequency bins of the spectrogram.
    t: Time bins of the spectrogram.
    
    Returns:
    dict: Extracted features.

    Features include:
    - 'normalized_std': Variance of the spectrogram, normalized by the mean.
    - 'period': Period of the signature, calculated as the mean of the time differences between peaks in the high frequency envelope (positive mirco Doppler).
    - 'offset': Offset of the signature, calculated as the difference between the mean (across time) of the high and low frequency envelopes (highest and lowest micro Doppler frequencies across time).
    - 'bandwidth': Bandwidth of the signature, calculated as the difference between the maximum and minimum micro Doppler frequencies.
    - 'bandwidth_wo_uD': Bandwidth of the signature without the micro Doppler, calculated as the difference between the minimum and maximum micro Doppler frequencies.
    - 'torso frequency':  The average frequency of the peak signal in strength over the time bins within the window.
    """
    #threshold_db = -10 # set threshold to -10 dB to filter out noise
    #spectrogram_thresh = np.where(spectrogram > threshold_db, spectrogram, 0) # Thresholding the spectrogram to remove noise

    peak_power_indices = np.argmax(spectrogram, axis=0)  # Index of max power in each time bin
    peak_freqs = f[peak_power_indices]  # Corresponding frequencies of the peak powers
    envelope_high = np.zeros(spectrogram.shape[1])  # Initialize envelope high
    envelope_low = np.zeros(spectrogram.shape[1])   # Initialize envelope low
    for i in range(spectrogram.shape[1]):
        non_zero_freq_indices = np.nonzero(spectrogram[:, i])[0]
        envelope_high[i] = f[np.max(non_zero_freq_indices)] # highest non-zero frequency in each time bin
        envelope_low[i] = f[np.min(non_zero_freq_indices)] # Minimum power in each time bin
    
    peaks, _ = scipy.signal.find_peaks(envelope_high)
    peak_times = t[peaks]

    features = {
        'normalized_std': np.std(spectrogram)/np.mean(spectrogram),
        'period': np.mean(np.diff(peak_times)),
        'offset': np.mean(envelope_high) - np.mean(envelope_low), 
        'bandwidth': np.max(envelope_high) - np.min(envelope_low),
        'bandwidth_wo_uD': np.min(envelope_high) - np.max(envelope_low),
        'torso frequency': np.mean(peak_freqs)
    }
    
    return features

def analyze_mean_and_std_of_features(features_df):
    """
    Analyze the mean and standard deviation of the extracted features after standardizing features.
    Parameters:
    features_df: DataFrame containing the extracted features with a 'label' column.
    """

    # Standardize features (excluding the label column)
    scaler = StandardScaler()
    features_only = features_df.drop(columns=['label'])
    features_std = scaler.fit_transform(features_only)
    features_std_df = pd.DataFrame(features_std, columns=features_only.columns)
    features_std_df['label'] = features_df['label'].values

    # Compute means and variances per label
    mean_per_label = features_std_df.groupby('label').mean()
    var_per_label = features_std_df.groupby('label').var()

    # Plot means with error bars (standard deviation)
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = mean_per_label.index
    x = range(len(labels))
    width = 0.12

    for i, feature in enumerate(mean_per_label.columns):
        if feature == 'label':
            continue
        means = mean_per_label[feature]
        stds = var_per_label[feature].apply(np.sqrt)
        ax.bar([xi + i*width for xi in x], means, width=width, yerr=stds, label=feature, capsize=4)

    ax.set_xticks([xi + width*2.5 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Standardized Feature Mean')
    ax.set_title('Standardized Feature Means and Std Dev per Class')
    ax.legend()
    plt.show()