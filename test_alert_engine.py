import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features
from alert.alert_engine import load_models, analyze_window

print("Loading models...")
models = load_models()
print("All 3 models loaded.\n")

# Load your real data
records  = parse_file('data/reference/app.log')
features = build_feature_vectors(records)
labels   = label_features(features)

SEQ_LEN = 10
feature_array = np.nan_to_num(features.values.astype(np.float32), nan=0.0)

print("="*60)
print("Testing on known anomaly windows")
print("="*60)

anomaly_indices = labels[labels == 1].index
tested = 0

for ts in anomaly_indices[:5]:   # test first 5 anomalies
    idx = features.index.get_loc(ts)
    if idx < SEQ_LEN:
        continue

    row       = feature_array[idx]
    seq       = feature_array[idx-SEQ_LEN:idx]
    feat_dict = features.iloc[idx].to_dict()

    result = analyze_window(row, feat_dict, models, seq)

    print(f"\nTimestamp : {ts}")
    print(f"Severity  : {result['severity']}")
    print(f"Reason    : {result['reason']}")
    print(f"Votes     : {result['votes']}/3")
    print(f"Hard rule : {result['hard_rule']}")
    print(f"Scores    : clf={result['scores']['classifier']['score']:.3f} "
          f"| ae={result['scores']['autoencoder']['score']:.1f} "
          f"| lstm={result['scores']['lstm']['score']:.3f}")
    tested += 1

print(f"\n{'='*60}")
print("Testing on 5 normal windows")
print("="*60)

# skip first 10 to ensure LSTM sequence is always available
normal_indices = labels[labels == 0].index
for ts in normal_indices[10:15]:
    idx = features.index.get_loc(ts)
    if idx < SEQ_LEN:
        continue

    row       = feature_array[idx]
    seq       = feature_array[idx-SEQ_LEN:idx]
    feat_dict = features.iloc[idx].to_dict()

    result = analyze_window(row, feat_dict, models, seq)
    print(f"\nTimestamp : {ts}")
    print(f"Severity  : {result['severity']}")
    print(f"Votes     : {result['votes']}/3")