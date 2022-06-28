from scipy.signal import butter, sosfiltfilt, sosfreqz


def butter_bandpass(low_freq_cutoff, high_freq_cutoff, fs, order=5):
    low = low_freq_cutoff
    high = high_freq_cutoff
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, low_freq_cutoff, high_freq_cutoff, fs, order=5):
    sos = butter_bandpass(low_freq_cutoff, high_freq_cutoff, fs, order)
    filtered = sosfiltfilt(sos, data)
    return filtered
