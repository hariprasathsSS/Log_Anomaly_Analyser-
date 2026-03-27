import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features
from models.lstm_detector import train_lstm

records  = parse_file('data/reference/app.log')
features = build_feature_vectors(records)
labels   = label_features(features)

print(f"Total windows : {len(features)}")
print(f"Normal        : {(labels==0).sum()}")
print(f"Anomaly       : {(labels==1).sum()}")

model, scaler = train_lstm(features, labels)