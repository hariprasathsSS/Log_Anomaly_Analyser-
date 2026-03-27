import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def build_autoencoder(input_dim: int) -> keras.Model:
    # Input
    inputs = keras.Input(shape=(input_dim,))

    # Encoder — compress 13 → 8 → 4
    x = keras.layers.Dense(8, activation='relu')(inputs)
    encoded = keras.layers.Dense(4, activation='relu')(x)

    # Decoder — reconstruct 4 → 8 → 13
    x = keras.layers.Dense(8, activation='relu')(encoded)
    decoded = keras.layers.Dense(input_dim, activation='linear')(x)

    model = keras.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(features, labels):
    X = features.values.astype(np.float32)
    y = labels.values.astype(np.float32)

    # Fill NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # KEY DIFFERENCE from classifier:
    # train ONLY on normal windows — anomalies never shown during training
    X_normal = X[y == 0]
    print(f"Training on {len(X_normal)} normal windows only")
    print(f"Anomaly windows held out: {(y==1).sum()} (used only for evaluation)")

    # Split normal data into train/val
    X_train, X_val = train_test_split(X_normal, test_size=0.1, random_state=42)

    model = build_autoencoder(X_train.shape[1])
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, X_train,   # input = output (reconstructing itself)
        epochs=50,
        batch_size=256,
        validation_data=(X_val, X_val),
        verbose=1
    )

    # Calculate reconstruction error on ALL data
    X_reconstructed = model.predict(X, verbose=0)
    reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=1)

    # Find threshold — 95th percentile of normal errors
    normal_errors  = reconstruction_errors[y == 0]
    anomaly_errors = reconstruction_errors[y == 1]
    threshold = np.percentile(normal_errors, 99)

    print(f"\nReconstruction error stats:")
    print(f"  Normal windows   — mean: {normal_errors.mean():.4f} · max: {normal_errors.max():.4f}")
    print(f"  Anomaly windows  — mean: {anomaly_errors.mean():.4f} · max: {anomaly_errors.max():.4f}")
    print(f"  Threshold (95th percentile of normal): {threshold:.4f}")

    # Evaluate
    y_pred = (reconstruction_errors > threshold).astype(int)
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred,
          target_names=['normal', 'anomaly'], zero_division=0))

    # Save
    model.save('models/autoencoder_model.keras')
    with open('models/autoencoder_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/autoencoder_threshold.pkl', 'wb') as f:
        pickle.dump(threshold, f)
    print("Saved → models/autoencoder_model.keras")

    return model, scaler, threshold, history