import numpy as np
import pickle
import torch
import tensorflow as tf
from tensorflow import keras
from models.lstm_detector import LSTMDetector

# ── Load all three models once at startup ─────────────────────────────

def load_models():
    # Keras classifier
    classifier = keras.models.load_model('models/classifier_model.keras')
    with open('models/scaler.pkl', 'rb') as f:
        clf_scaler = pickle.load(f)

    # TF autoencoder
    autoencoder = keras.models.load_model('models/autoencoder_model.keras')
    with open('models/autoencoder_scaler.pkl', 'rb') as f:
        ae_scaler = pickle.load(f)
    with open('models/autoencoder_threshold.pkl', 'rb') as f:
        ae_threshold = pickle.load(f)

    # PyTorch LSTM
    lstm = LSTMDetector(input_dim=13)
    lstm.load_state_dict(torch.load('models/lstm_model.pt', weights_only=True))
    lstm.eval()
    with open('models/lstm_scaler.pkl', 'rb') as f:
        lstm_scaler = pickle.load(f)

    return {
        'classifier'   : classifier,
        'clf_scaler'   : clf_scaler,
        'autoencoder'  : autoencoder,
        'ae_scaler'    : ae_scaler,
        'ae_threshold' : ae_threshold,
        'lstm'         : lstm,
        'lstm_scaler'  : lstm_scaler,
    }


# ── Hard rules — bypass ML entirely ──────────────────────────────────

def check_hard_rules(feature_row: dict) -> tuple[str, str] | None:
    """
    Returns (severity, reason) if a hard rule fires.
    Returns None if no hard rule matches.
    """
    bulk = feature_row.get('bulk_fetch_success_ratio', None)
    auth = feature_row.get('auth_failure_count', 0)
    err  = feature_row.get('internal_server_error_count', 0)
    okta = feature_row.get('okta_api_denied_count', 0)

    if bulk is not None and not np.isnan(bulk) and bulk == 0.0:
        return ('CRITICAL', 'Bulk fetch returned 0 successes')
    if auth >= 2:
        return ('CRITICAL', f'Auth failures: {int(auth)} in this window')
    if err >= 3:
        return ('HIGH', f'Repeated 500s: {int(err)} in this window')
    if okta >= 3:
        return ('HIGH', f'Okta access denied: {int(okta)} times')
    return None


# ── ML voting ────────────────────────────────────────────────────────

def run_models(feature_row: np.ndarray, models: dict,
               lstm_sequence: np.ndarray | None = None) -> dict:
    """
    Runs all 3 models on one feature row.
    lstm_sequence: last 10 rows including current (shape: 10 x 13)
    Returns individual scores and vote count.
    """
    row = feature_row.reshape(1, -1)
    votes = 0
    scores = {}

    # 1. Keras classifier
    clf_input = models['clf_scaler'].transform(row)
    clf_score = float(models['classifier'].predict(clf_input, verbose=0)[0][0])
    clf_vote  = 1 if clf_score > 0.5 else 0
    votes += clf_vote
    scores['classifier'] = {'score': clf_score, 'vote': clf_vote}

    # 2. TF autoencoder
    ae_input  = models['ae_scaler'].transform(row)
    ae_recon  = models['autoencoder'].predict(ae_input, verbose=0)
    ae_error  = float(np.mean(np.square(ae_input - ae_recon)))
    ae_vote   = 1 if ae_error > models['ae_threshold'] else 0
    votes += ae_vote
    scores['autoencoder'] = {'score': ae_error, 'vote': ae_vote}

    # 3. PyTorch LSTM (needs sequence)
    if lstm_sequence is not None:
        lstm_input = models['lstm_scaler'].transform(lstm_sequence)
        x = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = models['lstm'](x)
            lstm_score = float(torch.sigmoid(logit).item())
        lstm_vote = 1 if lstm_score > 0.5 else 0
        votes += lstm_vote
        scores['lstm'] = {'score': lstm_score, 'vote': lstm_vote}
    else:
        scores['lstm'] = {'score': None, 'vote': 0}

    scores['total_votes'] = votes
    return scores


# ── Final decision ────────────────────────────────────────────────────

def decide_alert(votes: int, hard_rule: tuple | None) -> tuple[str, str]:
    """Returns (severity, reason)"""
    if hard_rule:
        return hard_rule

    if votes == 0:
        return ('NORMAL',   'All models agree: normal window')
    if votes == 1:
        return ('LOW',      'One model flagged this window')
    if votes == 2:
        return ('HIGH',     'Two models agree: anomaly detected')
    return     ('CRITICAL', 'All three models agree: anomaly')


# ── Main entry point ──────────────────────────────────────────────────

def analyze_window(feature_row: np.ndarray,
                   feature_dict: dict,
                   models: dict,
                   lstm_sequence: np.ndarray | None = None) -> dict:
    """
    Full pipeline for one feature window.
    Returns a complete alert result dict.
    """
    # Fill NaN
    feature_row = np.nan_to_num(feature_row.astype(np.float32), nan=0.0)

    hard_rule = check_hard_rules(feature_dict)
    scores    = run_models(feature_row, models, lstm_sequence)
    severity, reason = decide_alert(scores['total_votes'], hard_rule)

    return {
        'severity'   : severity,
        'reason'     : reason,
        'votes'      : scores['total_votes'],
        'hard_rule'  : hard_rule is not None,
        'scores'     : scores,
    }