from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features
import pandas as pd

records  = parse_file('data/reference/app.log')
features = build_feature_vectors(records)
labels   = label_features(features)

# Get only anomaly windows
anomalies = features[labels == 1].copy()
anomalies['label'] = 1

# For each anomaly window, show WHY it was flagged
def why_flagged(row):
    reasons = []
    if row['bulk_fetch_success_ratio'] == 0.0:
        reasons.append('BULK FETCH FAILED (0 successes)')
    if row['auth_failure_count'] >= 1:
        reasons.append(f"AUTH FAILURE x{int(row['auth_failure_count'])}")
    if row['internal_server_error_count'] >= 3:
        reasons.append(f"500 ERRORS x{int(row['internal_server_error_count'])}")
    if row['okta_api_denied_count'] >= 1:
        reasons.append(f"OKTA DENIED x{int(row['okta_api_denied_count'])}")
    if row['has_celery_misuse'] == 1:
        reasons.append('CELERY MISUSE')
    if row['has_python_exception'] == 1:
        reasons.append('PYTHON EXCEPTION')
    if row['error_rate'] > 30:
        reasons.append(f"ERROR SPIKE ({int(row['error_rate'])} errors)")
    return ' | '.join(reasons)

anomalies['why'] = anomalies.apply(why_flagged, axis=1)

print(f"Total anomaly windows: {len(anomalies)}\n")
print("="*80)

# Now for each anomaly, also show the actual raw log lines from that minute
df_raw = pd.DataFrame(records)
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

for timestamp, row in anomalies.iterrows():
    print(f"\nTIMESTAMP : {timestamp}")
    print(f"WHY       : {row['why']}")
    print(f"FEATURES  : errors={int(row['error_rate'])} | "
          f"auth_fail={int(row['auth_failure_count'])} | "
          f"bulk_ratio={row['bulk_fetch_success_ratio']} | "
          f"celery={int(row['has_celery_misuse'])}")

    # Pull the actual raw lines from this exact minute
    window_start = timestamp
    window_end   = timestamp + pd.Timedelta(minutes=1)
    raw_lines = df_raw[
        (df_raw['timestamp'] >= window_start) &
        (df_raw['timestamp'] <  window_end)
    ]

    print(f"RAW LINES ({len(raw_lines)} lines in this minute):")
    for _, line in raw_lines.iterrows():
        marker = " <<<" if line['error_type'] else ""
        print(f"  [{line['level']}] {line['message'][:80]}{marker}")
    print("-"*80)