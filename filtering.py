import numpy as np
from scipy import signal


# buttorworth filter
def filter_series(series_in, cutoff_frequency):
  # sos = signal.butter(2, cutoff_frequency,'low', fs=30, output='sos')
  b,a = signal.butter(2, cutoff_frequency,'low', fs=30)
  # filtered = signal.sosfilt(sos, series_in)
  filtered = signal.filtfilt(b,a, series_in)

  return filtered

def hampel_filter(input_array, window_size, n_sigmas=3):

    k = 1.4826  # scale factor for Gaussian distribution
    new_array = input_array.copy()

    # helper lambda function
    MAD = lambda x: np.median(np.abs(x - np.median(x)))

    half_window = window_size // 2
    rolling_median = np.zeros_like(input_array)
    rolling_mad = np.zeros_like(input_array)

    for i in range(half_window, len(input_array) - half_window):
        window = input_array[i - half_window: i + half_window + 1]
        rolling_median[i] = np.median(window)
        rolling_mad[i] = k * MAD(window)

    diff = np.abs(input_array - rolling_median)

    indices = np.where(diff > (n_sigmas * rolling_mad))[0]
    indices_series = indices + half_window

    valid_indices = np.where((indices_series >= half_window) & (indices_series < len(input_array)))[0]
    indices_series = indices_series[valid_indices]
    indices = indices[valid_indices]

    new_array[indices_series] = rolling_median[indices]
    
    return new_array, indices_series