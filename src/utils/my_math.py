from numpy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def linear_slope(x, downsampling_factor=1):
    """
    Calculate the slope of a linear regression fitted to the provided series,
    with an option to downsample the series by a specified factor to reduce computation.

    Args:
        x (pd.Series): A window of the time series data.
        downsampling_factor (int): The factor by which to downsample the data.
                                   1 means no downsampling, 2 means take every second point, etc.

    Returns:
        float: The slope (coefficient) of the linear regression line fitted to the downsampled data within the window.
               This slope represents the trend direction and magnitude within the window.
               A positive slope indicates an increasing trend, while a negative slope indicates a decreasing trend.
    """
    # Downsample the series based on the downsampling factor
    if downsampling_factor > 1:
        x = x[::downsampling_factor]

    if len(x) < 2:
        return np.nan  # Not enough data to fit a linear model after downsampling

    y = x.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]


def spot_seasonal_cycles(data):
    """
    Identify the most dominant seasonal cycles using Fast Fourier Transform (FFT)
    and return their periods sorted by their strength (amplitude).

    The function accepts either a pandas Series or a NumPy array as input.

    Args:
        data (pd.Series or np.array): The time series data for which to identify dominant seasonal cycles.

    Returns:
        list: A list of tuples where each tuple contains the period of the cycle and its corresponding amplitude,
              sorted by amplitude in descending order.
    """
    # Convert the input to a NumPy array if it's a pandas Series
    if isinstance(data, pd.Series):
        values = data.values
    else:
        values = data

    # Perform FFT and get frequencies and amplitudes
    fft_values = np.fft.rfft(values)
    amplitudes = np.abs(fft_values)
    frequencies = np.fft.rfftfreq(len(values), d=1)  # Assuming daily data; adjust 'd' as needed

    # Exclude the zero frequency (trend component)
    frequencies = frequencies[1:]
    amplitudes = amplitudes[1:]

    # Calculate the periods from frequencies and sort by amplitude
    periods_amplitudes = [(1 / freq, amp) for freq, amp in zip(frequencies, amplitudes) if freq != 0]
    dominant_cycles = sorted(periods_amplitudes, key=lambda x: x[1], reverse=True)

    return dominant_cycles
