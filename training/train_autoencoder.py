import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.log_parser import parse_file
from ingestion.preprocessor import build_feature_vectors
from training.label_logs import label_features
from models.auto_encoder import train_autoencoder

records  = parse_file('data/reference/app.log')
features = build_feature_vectors(records)
labels   = label_features(features)

print(f"Total windows : {len(features)}")
print(f"Normal        : {(labels==0).sum()}")
print(f"Anomaly       : {(labels==1).sum()}")

model, scaler, threshold, history = train_autoencoder(features, labels)