import pandas as pd
import numpy as np

FEATURE_COLS = [
    'error_rate', 'warning_rate', 'auth_failure_count', 'forbidden_count',
    'internal_server_error_count', 'okta_api_denied_count',
    'bulk_fetch_success_ratio', 'entity_completion_count',
    'avg_completion_time_sec', 'celery_workers_active',
    'unique_error_modules', 'has_python_exception', 'has_celery_misuse',
]

def build_feature_vectors(records: list[dict], window_minutes: int = 1) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)

    freq = f'{window_minutes}min'

    features = pd.DataFrame()
    features['error_rate']   = (df['level'] == 'ERROR').resample(freq).sum()
    features['warning_rate'] = (df['level'] == 'WARNING').resample(freq).sum()

    features['auth_failure_count']          = (df['error_type'] == 'OktaTokenError').resample(freq).sum()
    features['forbidden_count']             = (df['error_type'] == 'Forbidden').resample(freq).sum()
    features['internal_server_error_count'] = (df['error_type'] == 'InternalError').resample(freq).sum()
    features['okta_api_denied_count']       = (df['error_type'] == 'OktaAccessDenied').resample(freq).sum()
    features['has_celery_misuse']           = (df['error_type'] == 'CeleryMisuse').resample(freq).sum().clip(0, 1)
    features['has_python_exception']        = (df['error_type'] == 'TypeError').resample(freq).sum().clip(0, 1)

    # bulk fetch success ratio per window
    def success_ratio(group):
        s = group.dropna()
        if len(s) == 0:
            return np.nan
        total = group['success_total'].dropna().sum()
        num   = group['success_num'].dropna().sum()
        return (num / total) if total > 0 else np.nan

    features['bulk_fetch_success_ratio'] = df.resample(freq).apply(
        lambda g: (g['success_num'].sum() / g['success_total'].sum())
        if g['success_total'].sum() > 0 else np.nan
    )

    features['entity_completion_count']  = df['entity'].notna().resample(freq).sum()
    features['avg_completion_time_sec']  = df['duration_sec'].resample(freq).mean()
    features['celery_workers_active']    = df['worker_id'].resample(freq).nunique()
    features['unique_error_modules']     = df[df['level'] == 'ERROR']['module'].resample(freq).nunique()

    # Fill most features with 0 — absence means nothing happened
    fill_zero = [c for c in FEATURE_COLS if c != 'bulk_fetch_success_ratio']
    features[fill_zero] = features[fill_zero].fillna(0)
    
    # bulk_fetch_success_ratio: leave NaN when no fetch ran in that window
    # Only fill 0 when a fetch ran but returned 0 successes
    # (NaN means "didn't run" — not an anomaly)
    features['bulk_fetch_success_ratio'] = features['bulk_fetch_success_ratio']
    return features[FEATURE_COLS]