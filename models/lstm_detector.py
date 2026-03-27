import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ── 1. Model definition ──────────────────────────────────────────────
class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, sequence, features)
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # take only the last time step's output
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step).squeeze(1)


# ── 2. Build sequences ───────────────────────────────────────────────
def build_sequences(X, y, seq_len=10):
    """
    Slide a window of seq_len minutes across the data.
    Each sequence is labelled 1 if ANY minute in it was anomalous.
    e.g. seq_len=10 means: look at 10 consecutive minutes → predict if anomalous
    """
    sequences, labels = [], []
    for i in range(len(X) - seq_len):
        seq   = X[i : i + seq_len]
        label = 1 if y[i : i + seq_len].max() > 0 else 0
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── 3. Train ─────────────────────────────────────────────────────────
def train_lstm(features, labels):
    X = features.values.astype(np.float32)
    y = labels.values.astype(np.float32)

    # Fill NaN + scale
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Build sequences
    SEQ_LEN = 10
    X_seq, y_seq = build_sequences(X, y, seq_len=SEQ_LEN)
    print(f"Sequences built : {len(X_seq)}")
    print(f"Anomaly seqs    : {y_seq.sum():.0f} ({y_seq.mean()*100:.1f}%)")


    # Stratified split — ensures anomalies appear in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq      # guarantees anomaly % is same in both splits
    )
    print(f"Train anomaly seqs : {y_train.sum():.0f}")
    print(f"Test anomaly seqs  : {y_test.sum():.0f}")

    # Class weight for imbalance
    n_normal  = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    pos_weight = torch.tensor([n_normal / max(n_anomaly, 1)], dtype=torch.float32)
    print(f"Pos weight      : {pos_weight.item():.1f}x")

    # DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # Model + loss + optimizer
    model     = LSTMDetector(input_dim=X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # The 5 steps of PyTorch training
            y_pred     = model(X_batch).squeeze()   # 1. predict
            loss       = criterion(y_pred, y_batch)  # 2. loss
            optimizer.zero_grad()                    # 5. clear old gradients
            loss.backward()                          # 3. backpropagate
            optimizer.step()                         # 4. update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/20 — loss: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test)
        logits   = model(X_test_t)
        # apply sigmoid manually here for prediction
        y_pred_t = torch.sigmoid(logits).numpy()

    for threshold in [0.3, 0.5]:
        y_pred = (y_pred_t > threshold).astype(int).flatten()
        print(f"\nThreshold = {threshold}")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(
            y_test, y_pred,
            target_names=['normal', 'anomaly'],
            labels=[0, 1],
            zero_division=0
        ))

    for threshold in [0.3, 0.5]:
        y_pred = (y_pred_t > threshold).astype(int)
        print(f"\nThreshold = {threshold}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred,
              target_names=['normal', 'anomaly'], zero_division=0))

    # Save
    torch.save(model.state_dict(), 'models/lstm_model.pt')
    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved → models/lstm_model.pt")

    return model, scaler