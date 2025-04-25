# Signal Quality Analyzer

A Python library for analyzing and visualizing the quality of time series signals, with a focus on physiological data like PPG (photoplethysmography).

## Overview

Signal Quality Analyzer provides tools to assess the quality of time series signals using autocorrelation-based methods. It includes an interactive dashboard built with Dash and Plotly for visual exploration of signal segments, allowing users to:

- Visualize raw and cleaned signal segments
- Compute and visualize autocorrelation to assess signal quality
- Search for high-quality signal segments
- Compare original and time-shifted signals

This library is particularly useful for researchers and developers working with physiological signals like PPG, ECG, EEG, etc., where signal quality assessment is crucial for reliable analysis.

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/signal-quality-analyzer.git

# Navigate to the repository
cd signal-quality-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- pandas
- numpy
- dash
- plotly
- neurokit2 (optional, for PPG/ECG processing)

## Quick Start

```python
import pandas as pd
import neurokit2 as nk
from signal_quality_analyzer import analyze_signal_quality

# Load your data
df = pd.read_csv("your_data.csv")

# Define a normalization function (optional)
def zscore_normalize(signal):
    import numpy as np
    return (signal - np.mean(signal)) / np.std(signal)

# Launch the dashboard
analyze_signal_quality(
    df,
    signal_col="your_signal_column",
    clean_func=nk.ppg_clean,  # or your custom cleaning function
    normalize_func=zscore_normalize,
    signal_label="Your Signal"
)
```

## Features

### Interactive Dashboard

The dashboard provides:

- Interactive time series visualization
- Signal segment selection slider
- Signal normalization option
- Quality assessment via autocorrelation
- Automatic search for high-quality segments
- Visual comparison of original and shifted signals

### Quality Assessment Method

The library assesses signal quality by:

1. Preparing a signal segment with appropriate padding
2. Cleaning the signal using a provided cleaning function
3. Computing autocorrelation between the core signal and time-shifted versions
4. Identifying peaks in the autocorrelation within a configurable lag window
5. Using the maximum correlation value as a quality metric

## API Reference

### Main Functions

#### `analyze_signal_quality(df, signal_col, clean_func, **kwargs)`

Convenience function to create and run a signal quality dashboard.

**Parameters:**
- `df`: DataFrame containing the signal
- `signal_col`: Column name containing the signal
- `clean_func`: Function to clean the signal
- `**kwargs`: Additional arguments to pass to SignalQualityDashboard

### SignalQualityDashboard Class

#### `SignalQualityDashboard(df, signal_col, clean_func, **kwargs)`

Creates an interactive dashboard for signal quality assessment.

**Parameters:**
- `df`: DataFrame containing the signal
- `signal_col`: Column name containing the signal
- `clean_func`: Function to clean the signal
- `signal_label`: Label for the signal (default: "Signal")
- `normalize_func`: Optional function to normalize the signal
- `sampling_rate`: Sampling rate in Hz (default: 186)
- `segment_duration_sec`: Duration of segment in seconds (default: 10)
- `quality_lag_low`: Lower bound for quality assessment lag, in seconds (default: 0.5)
- `quality_lag_high`: Upper bound for quality assessment lag, in seconds (default: 1.5)
- `quality_threshold`: Threshold for good quality (default: 0.6)
- `max_lag_sec`: Maximum lag in seconds for autocorrelation (default: 3)
- `clean_padding_ratio`: Ratio of padding to add before cleaning (default: 0.2)
- `delta_padding_sec`: Padding in seconds for delta calculation (default: 1.6)
- `port`: Port for the Dash app (default: 32423)

#### `run(debug=True)`

Runs the dashboard.

### Utility Functions

#### `prepare_signal_segment(df, start_idx, signal_col, clean_func, segment_duration_sec, sampling_rate, max_delta_sec, clean_padding_ratio)`

Prepares a segment of signal data for analysis.

#### `compute_autocorr_extended(cleaned_core, cleaned_extended, max_lag_sec, sampling_rate)`

Computes autocorrelation between core signal and extended signal.

## Examples

### Basic Example

```python
import pandas as pd
import neurokit2 as nk
from signal_quality_analyzer import analyze_signal_quality

# Load PPG data
ppg_df = pd.read_csv("ppg_data.csv")

# Run the dashboard with default parameters
analyze_signal_quality(
    ppg_df,
    signal_col="ppg",
    clean_func=nk.ppg_clean,
    signal_label="PPG"
)
```

### Advanced Example

```python
import pandas as pd
import numpy as np
import neurokit2 as nk
from signal_quality_analyzer import SignalQualityDashboard

# Load data
ecg_df = pd.read_csv("ecg_data.csv")

# Custom cleaning function
def custom_ecg_clean(signal, sampling_rate):
    # Apply bandpass filter
    filtered = nk.signal_filter(signal, lowcut=0.5, highcut=40, sampling_rate=sampling_rate)
    # Apply additional processing if needed
    return filtered

# Custom normalization
def robust_normalize(signal):
    q25, q75 = np.percentile(signal, [25, 75])
    iqr = q75 - q25
    return (signal - q25) / iqr

# Create and run dashboard with custom parameters
dashboard = SignalQualityDashboard(
    ecg_df,
    signal_col="ecg",
    clean_func=custom_ecg_clean,
    normalize_func=robust_normalize,
    signal_label="ECG",
    sampling_rate=250,
    segment_duration_sec=5,
    quality_lag_low=0.2,
    quality_lag_high=1.0,
    quality_threshold=0.7
)
dashboard.run()
```

## Customization

### Custom Cleaning Functions

The library can work with any cleaning function that follows this signature:

```python
def custom_clean(signal, sampling_rate=None, **kwargs):
    # Process the signal
    cleaned_signal = ...
    return cleaned_signal
```

### Custom Normalization

You can provide any normalization function with this signature:

```python
def custom_normalize(signal):
    # Normalize the signal
    normalized = ...
    return normalized
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Neurokit2](https://github.com/neuropsychology/NeuroKit) for signal processing tools
- [Dash](https://dash.plotly.com/) for the interactive visualization framework
- [Plotly](https://plotly.com/) for the plotting library