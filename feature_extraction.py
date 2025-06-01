import numpy as np
import scipy 

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
    - 'period': Period of the spectrogram, calculated as the mean of the time differences between peaks in the high frequency envelope (positive mirco Doppler).
    - 'offset': Offset of the spectrogram, calculated as the difference between the mean (across time) of the high and low frequency envelopes (highest and lowest micro Doppler frequencies across time).
    - 'bandwidth': Bandwidth of the spectrogram, calculated as the difference between the maximum and minimum micro Doppler frequencies.
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
        'torso frequency': np.mean(peak_freqs)
    }
    
    return features