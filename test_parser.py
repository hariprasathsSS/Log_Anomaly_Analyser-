from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features

records = parse_file('data/reference/app.log')
print(f"Parsed {len(records)} log lines")

features = build_feature_vectors(records)
labels   = label_features(features)

print(f"\nFeature matrix shape : {features.shape}")
print(f"Total windows        : {len(labels)}")
print(f"Normal windows  (0)  : {(labels == 0).sum()}")
print(f"Anomaly windows (1)  : {(labels == 1).sum()}")
print(f"Anomaly %            : {labels.mean()*100:.1f}%")

# Show some confirmed anomaly windows
print("\nSample anomaly windows:")
anomaly_features = features[labels == 1]
print(anomaly_features[['error_rate','auth_failure_count',
                         'bulk_fetch_success_ratio',
                         'has_celery_misuse']].head(10))