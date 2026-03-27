import pandas as pd
import numpy as np

def label_features(features: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of 0 (normal) or 1 (anomaly) for each row.
    Rules are derived directly from what we know about app.log.
    """
    labels = pd.Series(0, index=features.index, name='label')

    # RIGHT — only fire when a fetch actually ran AND got 0 successes
    bulk = features['bulk_fetch_success_ratio']
    labels[(bulk.notna()) & (bulk == 0.0)] = 1

    # Auth failures are rare in normal ops — any occurrence is anomaly
    labels[features['auth_failure_count'] >= 1] = 1

    # Repeated internal server errors in a window
    labels[features['internal_server_error_count'] >= 3] = 1

    # Okta access denied
    labels[features['okta_api_denied_count'] >= 1] = 1

    # Celery misuse is always a misconfiguration
    labels[features['has_celery_misuse'] == 1] = 1

    # Python exceptions present
    labels[features['has_python_exception'] == 1] = 1

    # High error spike (error_rate > 20 in a single minute is abnormal)
    labels[features['error_rate'] > 30] = 1

    return labels