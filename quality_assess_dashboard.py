"""
Signal Quality Analyzer

A module for analyzing and visualizing the quality of time series signals, 
with a focus on physiological data like PPG (photoplethysmography).

Features:
- Signal segment preparation with padding for analysis
- Autocorrelation-based quality assessment
- Interactive dashboard for visual exploration
- Quality threshold-based segment search

Example usage:
    from signal_quality_analyzer import analyze_signal_quality
    import pandas as pd
    import neurokit2 as nk
    
    # Load data
    ppg_df = pd.read_csv("data.csv")
    
    # Optional: Define a normalization function
    def zscore_normalize(signal):
        import numpy as np
        return (signal - np.mean(signal)) / np.std(signal)
    
    # Run the analyzer
    analyze_signal_quality(
        ppg_df,
        signal_col="ppg0",
        clean_func=nk.ppg_clean,
        normalize_func=zscore_normalize,
        signal_label="PPG"
    )
"""

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
from datetime import timedelta


def prepare_signal_segment(df, start_idx, signal_col, clean_func,
                           segment_duration_sec, sampling_rate, max_delta_sec, clean_padding_ratio):
    """
    Prepare a segment of signal data for analysis.
    
    Args:
        df: DataFrame containing the signal
        start_idx: Starting index for the segment
        signal_col: Column name containing the signal
        clean_func: Function to clean the signal
        segment_duration_sec: Duration of segment in seconds
        sampling_rate: Sampling rate in Hz
        max_delta_sec: Maximum delta time in seconds for autocorrelation
        clean_padding_ratio: Ratio of padding to add before cleaning
        
    Returns:
        Tuple of (raw_signal, cleaned_core_signal, cleaned_extended_signal, time_array)
    """
    segment_length = segment_duration_sec * sampling_rate
    clean_padding = int(clean_padding_ratio * segment_length)
    delta_padding = int(max_delta_sec * sampling_rate)

    total_padding_before = clean_padding
    total_padding_after = clean_padding + delta_padding

    start = max(0, start_idx - total_padding_before)
    end = start_idx + segment_length + total_padding_after
    segment_df = df.iloc[start:end]
    signal_raw = segment_df[signal_col].values

    signal_cleaned_full = clean_func(signal_raw, sampling_rate=sampling_rate)
    plot_start = total_padding_before
    plot_end = plot_start + segment_length

    signal_cleaned_core = signal_cleaned_full[plot_start:plot_end]
    signal_cleaned_extended = signal_cleaned_full[plot_start:plot_end + delta_padding]
    signal_raw_trimmed = signal_raw[plot_start:plot_end]

    start_time_ns = df.iloc[start_idx]["time"]
    start_time = pd.to_datetime(start_time_ns)
    time = [start_time + timedelta(seconds=i / sampling_rate) for i in range(segment_length)]

    return signal_raw_trimmed, signal_cleaned_core, signal_cleaned_extended, time


def compute_autocorr_extended(cleaned_core, cleaned_extended, max_lag_sec, sampling_rate):
    """
    Compute autocorrelation between core signal and extended signal.
    
    Args:
        cleaned_core: Core cleaned signal
        cleaned_extended: Extended cleaned signal (includes core plus padding)
        max_lag_sec: Maximum lag in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (normalized_correlations, lag_times)
    """
    def safe_correlation(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    max_shift = int(max_lag_sec * sampling_rate)
    corrs = []
    for lag in range(max_shift):
        if lag + len(cleaned_core) > len(cleaned_extended):
            corrs.append(0)
        else:
            corr = safe_correlation(cleaned_core, cleaned_extended[lag:lag + len(cleaned_core)])
            corrs.append(corr)

    corrs = np.nan_to_num(np.array(corrs))
    normed_corrs = corrs / np.max(corrs) if np.max(corrs) > 0 else corrs
    return normed_corrs, np.arange(len(corrs)) / sampling_rate


def _setup_layout(app, df, signal_label, normalize_func, sampling_rate, 
                segment_duration_sec, quality_threshold):
    """Set up the dashboard layout."""
    normalize_checkbox = dcc.Checklist(
        options=[{"label": "Normalize Signal", "value": "yes"}],
        id="normalize-signal", value=[]
    ) if normalize_func else html.Div()

    app.layout = html.Div([
        html.H2(f"{signal_label} Quality Dashboard"),

        dcc.Graph(id="signal-plot"),

        html.Label("Select Segment"),
        dcc.Slider(id='segment-slider', min=0,
                  max=len(df) - sampling_rate * segment_duration_sec,
                  step=sampling_rate * segment_duration_sec,
                  value=0,
                  tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Checklist(
            options=[
                {"label": "Show Raw", "value": "raw"},
                {"label": "Show Cleaned", "value": "cleaned"}
            ],
            value=["cleaned"],
            id="show-options"
        ),
        normalize_checkbox,

        html.Div([
            html.Label("Quality Threshold"),
            dcc.Input(id="quality-input", type="number", value=quality_threshold, step=0.01)
        ], style={"marginTop": "15px"}),

        html.Div([
            html.Button("⬅️ Search Left", id="search-left", n_clicks=0,
                        style={"marginRight": "10px", "backgroundColor": "#f0f0f0", "padding": "8px 16px",
                              "border": "1px solid #ccc", "borderRadius": "5px"}),
            html.Button("Search Right ➡️", id="search-right", n_clicks=0,
                        style={"marginRight": "10px", "backgroundColor": "#f0f0f0", "padding": "8px 16px",
                              "border": "1px solid #ccc", "borderRadius": "5px"}),
            html.Button("✔️ Assess Quality", id="assess-btn", n_clicks=0,
                        style={"backgroundColor": "#DFF0D8", "padding": "8px 16px",
                              "border": "1px solid #ccc", "borderRadius": "5px"})
        ], style={"marginTop": "10px"}),

        html.Div(id="search-status", style={"color": "#888", "marginTop": "10px"}),

        html.Div(id="quality-output", style={"marginTop": "15px", "fontWeight": "bold"}),

        dcc.Graph(id="autocorr-plot"),
        dcc.Graph(id="shifted-plot")
    ])


def _setup_callbacks(app, df, signal_col, clean_func, signal_label, normalize_func,
                   sampling_rate, segment_duration_sec, quality_lag_low, 
                   quality_lag_high, delta_padding_sec, clean_padding_ratio, max_lag_sec):
    """Set up the dashboard callbacks."""
    
    @app.callback(
        Output("signal-plot", "figure"),
        Input("segment-slider", "value"),
        Input("show-options", "value"),
        Input("normalize-signal", "value") if normalize_func else State("segment-slider", "value")
    )
    def update_signal_plot(start_idx, show_opts, normalize_signal):
        raw, cleaned_core, _, time = prepare_signal_segment(
            df, start_idx, signal_col, clean_func,
            segment_duration_sec, sampling_rate,
            delta_padding_sec, clean_padding_ratio
        )
        
        if normalize_func and "yes" in normalize_signal:
            raw = normalize_func(raw)
            cleaned_core = normalize_func(cleaned_core)

        fig = go.Figure()
        if "raw" in show_opts:
            fig.add_trace(go.Scatter(x=time, y=raw, mode='lines', name='Raw', line=dict(color='gray')))
        if "cleaned" in show_opts:
            fig.add_trace(go.Scatter(x=time, y=cleaned_core, mode='lines', name='Cleaned', line=dict(color='blue')))
        fig.update_layout(title="Signal Segment", xaxis_title="Time", yaxis_title=signal_label)
        return fig

    @app.callback(
        [Output("autocorr-plot", "figure"),
         Output("shifted-plot", "figure"),
         Output("quality-output", "children")],
        Input("assess-btn", "n_clicks"),
        State("segment-slider", "value"),
        State("normalize-signal", "value") if normalize_func else State("segment-slider", "value"),
        State("quality-input", "value")
    )
    def assess_quality(n, start_idx, normalize_signal, quality_thresh):
        raw, cleaned_core, cleaned_ext, time = prepare_signal_segment(
            df, start_idx, signal_col, clean_func,
            segment_duration_sec, sampling_rate,
            delta_padding_sec, clean_padding_ratio
        )

        if normalize_func and "yes" in normalize_signal:
            cleaned_core = normalize_func(cleaned_core)
            cleaned_ext = normalize_func(cleaned_ext)

        autocorr, lags = compute_autocorr_extended(
            cleaned_core, cleaned_ext,
            max_lag_sec, sampling_rate
        )

        min_idx = int(quality_lag_low * sampling_rate)
        max_idx = int(quality_lag_high * sampling_rate)
        best_idx = np.argmax(autocorr[min_idx:max_idx]) + min_idx
        best_lag = lags[best_idx]
        best_score = float(np.nan_to_num(autocorr[best_idx]))

        auto_fig = go.Figure()
        auto_fig.add_trace(go.Scatter(x=lags, y=autocorr, mode='lines', name='Autocorrelation'))
        auto_fig.add_trace(go.Scatter(x=[best_lag], y=[best_score], mode='markers',
                                     marker=dict(color='red', size=10), name='Max Corr'))
        auto_fig.update_layout(title="Autocorrelation", xaxis_title="Lag (s)", yaxis_title="Normalized Corr")

        shifted = cleaned_ext[best_idx:best_idx + len(cleaned_core)]
        shift_fig = go.Figure()
        shift_fig.add_trace(go.Scatter(x=time, y=cleaned_core, mode='lines', name='Cleaned'))
        shift_fig.add_trace(go.Scatter(x=time, y=shifted, mode='lines', name=f'Shifted (Δt={best_lag:.2f}s)',
                                      line=dict(color='red')))
        shift_fig.update_layout(title="Shifted Signal", xaxis_title="Time", yaxis_title=signal_label)

        quality_text = f"Max correlation in [{quality_lag_low}s–{quality_lag_high}s] = {best_score:.2f} → " + \
                      ("Good ✅" if best_score >= quality_thresh else "Poor ❌")
        return auto_fig, shift_fig, quality_text

    @app.callback(
        Output("segment-slider", "value"),
        Output("search-status", "children"),
        Input("search-left", "n_clicks"),
        Input("search-right", "n_clicks"),
        State("segment-slider", "value"),
        State("normalize-signal", "value") if normalize_func else State("segment-slider", "value"),
        State("quality-input", "value")
    )
    def search_quality_segment(n_left, n_right, current_idx, normalize_signal, quality_thresh):
        triggered = ctx.triggered_id
        if triggered not in ["search-left", "search-right"]:
            return current_idx, ""

        direction = -1 if triggered == "search-left" else 1
        step = sampling_rate * segment_duration_sec
        new_idx = current_idx + step * direction

        while 0 <= new_idx <= len(df) - sampling_rate * segment_duration_sec:
            raw, cleaned_core, cleaned_ext, _ = prepare_signal_segment(
                df, new_idx, signal_col, clean_func,
                segment_duration_sec, sampling_rate,
                delta_padding_sec, clean_padding_ratio
            )

            if normalize_func and "yes" in normalize_signal:
                cleaned_core = normalize_func(cleaned_core)
                cleaned_ext = normalize_func(cleaned_ext)

            autocorr, _ = compute_autocorr_extended(
                cleaned_core, cleaned_ext,
                max_lag_sec, sampling_rate
            )

            min_idx = int(quality_lag_low * sampling_rate)
            max_idx = int(quality_lag_high * sampling_rate)
            best_score = np.nan_to_num(np.max(autocorr[min_idx:max_idx]))

            if best_score >= quality_thresh:
                return new_idx, ""

            new_idx += step * direction

        return current_idx, "No segment found with quality above threshold."


def analyze_signal_quality(df, signal_col, clean_func, **kwargs):
    """
    Create and run a signal quality dashboard.
    
    Args:
        df: DataFrame containing the signal
        signal_col: Column name containing the signal
        clean_func: Function to clean the signal
        **kwargs: Additional parameters, supporting:
            - signal_label: Label for the signal (default "Signal")
            - normalize_func: Optional function to normalize the signal
            - sampling_rate: Sampling rate in Hz (default 186)
            - segment_duration_sec: Duration of segment in seconds (default 10)
            - quality_lag_low: Lower bound for quality assessment lag, in seconds (default 0.5)
            - quality_lag_high: Upper bound for quality assessment lag, in seconds (default 1.5)
            - quality_threshold: Threshold for good quality (default 0.6)
            - max_lag_sec: Maximum lag in seconds for autocorrelation (default 3)
            - clean_padding_ratio: Ratio of padding to add before cleaning (default 0.2)
            - delta_padding_sec: Padding in seconds for delta calculation (default 1.6)
            - port: Port for the Dash app (default 32423)
            - debug: Run Dash app in debug mode (default True)
        
    Returns:
        None
    """
    # Extract parameters from kwargs with defaults
    signal_label = kwargs.get("signal_label", "Signal")
    normalize_func = kwargs.get("normalize_func", None)
    sampling_rate = kwargs.get("sampling_rate", 186)
    segment_duration_sec = kwargs.get("segment_duration_sec", 10)
    quality_lag_low = kwargs.get("quality_lag_low", 0.5)
    quality_lag_high = kwargs.get("quality_lag_high", 1.5)
    quality_threshold = kwargs.get("quality_threshold", 0.6)
    max_lag_sec = kwargs.get("max_lag_sec", 3)
    clean_padding_ratio = kwargs.get("clean_padding_ratio", 0.2)
    delta_padding_sec = kwargs.get("delta_padding_sec", 1.6)
    port = kwargs.get("port", 32423)
    debug = kwargs.get("debug", False)
    
    # Create the Dash app
    app = Dash(__name__)
    
    # Setup layout and callbacks
    _setup_layout(app, df, signal_label, normalize_func, sampling_rate, 
                segment_duration_sec, quality_threshold)
    
    _setup_callbacks(app, df, signal_col, clean_func, signal_label, normalize_func,
                   sampling_rate, segment_duration_sec, quality_lag_low, quality_lag_high, 
                   delta_padding_sec, clean_padding_ratio, max_lag_sec)
    
    # Run the app
    app.run(debug=debug, port=port)


# Example usage (only executes when run as a script, not when imported)
if __name__ == '__main__':
    import neurokit2 as nk
    
    # Load data
    ppg_df = pd.read_csv("48.csv", nrows=10000000)
    ppg_df = ppg_df[-100000:]
    
    # Define normalization function
    def zscore_normalize(signal):
        return (signal - np.mean(signal)) / np.std(signal)
    
    # Run the dashboard
    analyze_signal_quality(
        ppg_df,
        signal_col="ppg0",
        clean_func=nk.ppg_clean,
        normalize_func=zscore_normalize,
        signal_label="PPG"
    )