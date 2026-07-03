"""
Anomaly Detection System for process sensor time-series data.

Designed for tag-based DCS/SCADA-style logs (Time + one column per tag,
1-second resolution) like 2040-TT-6-B06K / 2040-FIT-6-B06U / 2040-PT-6-B06U.

Detectors included:
  1. Sensor dropout   - value == 0 (or NaN) across all tags simultaneously
  2. Flatline / stuck sensor - value doesn't change for N consecutive seconds
  3. Time gap         - missing samples in the timestamp stream
  4. Statistical outlier - rolling z-score per tag (catches spikes/drift)
  5. Rate-of-change spike - |delta| between consecutive samples too large
  6. Multivariate anomaly - Isolation Forest across all tags jointly

Usage:
    python anomaly_detection.py <input.xlsx> [--out-dir DIR]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

TAG_COLUMNS = None  # auto-detected: all columns except 'Time'

# ---- Tunable parameters -----------------------------------------------
ROLLING_WINDOW = 300          # seconds, for rolling mean/std (z-score)
Z_THRESHOLD = 4.5             # |z| above this = statistical outlier
FLATLINE_SECONDS = 120        # value unchanged for this long = stuck sensor
GAP_THRESHOLD_SECONDS = 60    # missing time gap larger than this = data gap
ROC_Z_THRESHOLD = 7.0         # z-score of first-difference = spike
IFOREST_CONTAMINATION = 0.002 # expected fraction of multivariate anomalies
MERGE_GAP_SECONDS = 5         # merge same type/tag events this close together


def load_data(path):
    df = pd.read_excel(path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    global TAG_COLUMNS
    TAG_COLUMNS = [c for c in df.columns if c != 'Time']
    return df


def detect_dropouts(df):
    """All tags reading exactly 0 (or NaN) at the same timestamp = comms/sensor dropout."""
    all_zero = (df[TAG_COLUMNS].fillna(0) == 0).all(axis=1)
    return flag_runs(df, all_zero, 'dropout', severity='high')


def detect_flatline(df):
    """A tag stuck at an identical value for longer than FLATLINE_SECONDS."""
    events = []
    for col in TAG_COLUMNS:
        same_as_prev = df[col].diff().eq(0)
        run_id = (~same_as_prev).cumsum()
        run_lengths = same_as_prev.groupby(run_id).transform('sum') + 1
        stuck = run_lengths >= FLATLINE_SECONDS
        stuck = stuck & df[col].notna()
        events.append(flag_runs(df, stuck, f'flatline:{col}', severity='medium'))
    return pd.concat(events, ignore_index=True) if events else empty_events()


def detect_time_gaps(df):
    dt = df['Time'].diff().dt.total_seconds()
    gap_mask = dt > GAP_THRESHOLD_SECONDS
    rows = []
    for idx in df.index[gap_mask]:
        rows.append({
            'type': 'time_gap',
            'tag': 'Time',
            'severity': 'high' if dt.loc[idx] > 600 else 'medium',
            'start': df['Time'].iloc[idx - 1],
            'end': df['Time'].iloc[idx],
            'duration_s': dt.loc[idx],
            'detail': f'{dt.loc[idx]:.0f}s gap in data stream',
        })
    return pd.DataFrame(rows) if rows else empty_events()


def detect_statistical_outliers(df):
    events = []
    for col in TAG_COLUMNS:
        series = df[col]
        roll_mean = series.rolling(ROLLING_WINDOW, min_periods=30, center=True).mean()
        roll_std = series.rolling(ROLLING_WINDOW, min_periods=30, center=True).std()
        z = (series - roll_mean) / roll_std.replace(0, np.nan)
        outlier = z.abs() > Z_THRESHOLD
        outlier = outlier & series.notna()
        ev = flag_runs(df, outlier, f'statistical_outlier:{col}', severity='medium')
        if not ev.empty:
            ev['detail'] = ev.apply(
                lambda r: f'max |z|={z.loc[(df.Time >= r.start) & (df.Time <= r.end)].abs().max():.1f}',
                axis=1)
        events.append(ev)
    return pd.concat(events, ignore_index=True) if events else empty_events()


def detect_roc_spikes(df):
    events = []
    for col in TAG_COLUMNS:
        diff = df[col].diff()
        roll_std = diff.rolling(ROLLING_WINDOW, min_periods=30, center=True).std()
        z = diff / roll_std.replace(0, np.nan)
        spike = z.abs() > ROC_Z_THRESHOLD
        spike = spike & df[col].notna()
        events.append(flag_runs(df, spike, f'roc_spike:{col}', severity='low'))
    return pd.concat(events, ignore_index=True) if events else empty_events()


def detect_multivariate(df):
    X = df[TAG_COLUMNS].copy()
    valid = X.notna().all(axis=1)
    X_valid = X[valid]
    if len(X_valid) < 100:
        return empty_events()
    clf = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=42, n_jobs=-1)
    pred = clf.fit_predict(X_valid)
    mask_valid = pd.Series(pred == -1, index=X_valid.index)
    mask = pd.Series(False, index=df.index)
    mask.loc[mask_valid.index] = mask_valid
    return flag_runs(df, mask, 'multivariate_anomaly', severity='medium')


def flag_runs(df, bool_mask, event_type, severity):
    """Collapse a boolean mask into contiguous start/end runs, then merge
    runs of the same type/tag separated only by a short gap (avoids
    fragmenting one real event into many rows of flickering noise)."""
    bool_mask = bool_mask.fillna(False)
    if not bool_mask.any():
        return empty_events()
    run_id = (bool_mask != bool_mask.shift()).cumsum()
    rows = []
    for _, grp in df[bool_mask].groupby(run_id[bool_mask]):
        rows.append({
            'type': event_type.split(':')[0],
            'tag': event_type.split(':')[1] if ':' in event_type else 'ALL',
            'severity': severity,
            'start': grp['Time'].iloc[0],
            'end': grp['Time'].iloc[-1],
            'duration_s': (grp['Time'].iloc[-1] - grp['Time'].iloc[0]).total_seconds() + 1,
            'n_points': len(grp),
            'detail': '',
        })
    events = pd.DataFrame(rows)
    return merge_nearby(events)


def merge_nearby(events):
    if events.empty:
        return events
    events = events.sort_values('start').reset_index(drop=True)
    merged = [events.iloc[0].to_dict()]
    for _, row in events.iloc[1:].iterrows():
        last = merged[-1]
        gap = (row['start'] - last['end']).total_seconds()
        if gap <= MERGE_GAP_SECONDS:
            last['end'] = row['end']
            last['duration_s'] = (last['end'] - last['start']).total_seconds() + 1
            last['n_points'] += row['n_points']
        else:
            merged.append(row.to_dict())
    return pd.DataFrame(merged)


def empty_events():
    return pd.DataFrame(columns=['type', 'tag', 'severity', 'start', 'end',
                                  'duration_s', 'n_points', 'detail'])


def run_all_detectors(df):
    results = {
        'dropout': detect_dropouts(df),
        'flatline': detect_flatline(df),
        'time_gap': detect_time_gaps(df),
        'statistical_outlier': detect_statistical_outliers(df),
        'roc_spike': detect_roc_spikes(df),
        'multivariate_anomaly': detect_multivariate(df),
    }
    combined = pd.concat(results.values(), ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values('start').reset_index(drop=True)
    return combined, results


def summarize(combined):
    if combined.empty:
        return pd.DataFrame()
    return (combined.groupby(['type', 'severity'])
            .agg(count=('type', 'size'), total_duration_s=('duration_s', 'sum'))
            .reset_index()
            .sort_values('count', ascending=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()

    df = load_data(args.input)
    combined, per_detector = run_all_detectors(df)
    summary = summarize(combined)

    combined.to_csv(f'{args.out_dir}/anomalies.csv', index=False)
    summary.to_csv(f'{args.out_dir}/anomaly_summary.csv', index=False)

    print(summary.to_string(index=False))
    print(f'\nTotal anomaly events: {len(combined)}')
