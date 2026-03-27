from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features
from models.classifiers import train_classifier

records  = parse_file('data/reference/app.log')
features = build_feature_vectors(records)
labels   = label_features(features)

print(f"Training on {len(features)} windows")
print(f"Anomalies : {labels.sum()} ({labels.mean()*100:.1f}%)")

model, scaler, history = train_classifier(features, labels)