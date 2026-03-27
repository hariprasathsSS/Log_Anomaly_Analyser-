# Log Anomaly Detector

A production-grade ML anomaly detection system built on real Django and Celery log data. Uses three independent machine learning models — a Keras binary classifier, a TensorFlow autoencoder, and a PyTorch LSTM — combined through an ensemble alert engine to detect anomalies in application logs with high confidence and minimal false alarms.

---

## What This System Does

Your application writes thousands of log lines every minute. Most are normal. A few indicate real problems — failed bulk fetches, Okta authentication errors, Celery misconfigurations, repeated 500s on the same endpoint. This system reads those logs continuously, converts them into numeric feature vectors, runs them through three ML models, and fires alerts when something is genuinely wrong.

The key design insight: a single log line tells you almost nothing. What matters is the pattern across many lines in a short window of time. This system groups lines into 1-minute buckets, extracts 13 meaningful signals from each bucket, and lets the models decide whether that minute looked normal or anomalous.

---

## Project Structure

```
log-anomaly-detector/
├── ingestion/
│   ├── __init__.py
│   ├── log_parser.py         # parses Django + Celery log formats
│   ├── preprocessor.py       # builds 13-feature vectors from parsed lines
│   └── log_reader.py         # tails live log files for real-time monitoring
│
├── models/
│   ├── __init__.py
│   ├── classifiers.py        # Keras binary classifier
│   ├── autoencoder.py        # TensorFlow autoencoder
│   └── lstm_detector.py      # PyTorch LSTM sequence detector
│
├── training/
│   ├── label_logs.py         # rule-based auto-labeller
│   ├── train_classifier.py   # trains the Keras classifier
│   ├── train_autoencoder.py  # trains the TF autoencoder
│   └── train_lstm.py         # trains the PyTorch LSTM
│
├── alert/
│   ├── __init__.py
│   └── alert_engine.py       # combines all 3 models into final alerts
│
├── data/
│   └── reference/
│       └── app.log           # your training log file
│
├── main.py                   # entry point for live monitoring
├── demo_replayer.py          # replays app.log as if it's live
├── test_alert_engine.py      # end-to-end smoke test
├── test_parse.py             # verifies parser and feature extraction
├── verify_anomalies.py       # shows raw log lines behind each anomaly label
└── requirements.txt
```

---

## Quick Start

```bash
# 1. create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. install dependencies
pip install torch tensorflow scikit-learn pandas numpy python-dateutil imbalanced-learn

# 3. place your log file
cp /path/to/app.log data/reference/app.log

# 4. verify parsing works
python test_parse.py

# 5. train all three models
python -m training.train_classifier
python -m training.train_autoencoder
python -m training.train_lstm

# 6. run end-to-end test
python test_alert_engine.py

# 7. start live monitoring
python main.py
```

---

## Phase 1 — Ingestion

### `ingestion/log_parser.py`

**Why this file exists:** Your application produces two completely different log formats from two different systems. Django logs look like this:

```
2026-01-20 15:55:02,475 | INFO | base.add_job:507 | Adding job tentatively...
```

Celery worker logs look like this:

```
[2026-01-20 15:11:44,923: INFO/ForkPoolWorker-5] [ENTITY: users] Starting processing...
```

Same information, completely different structure. A single regex cannot handle both. The parser defines two separate patterns and tries each one on every line.

**Key functions:**

`parse_line(line: str) -> dict | None`

Tries the Django regex first, then the Celery regex. If neither matches (stack traces, blank lines, separator bars), returns None and the line is discarded. If a match is found, extracts all useful fields and returns a clean dictionary:

```python
{
    'timestamp'    : datetime(2026, 1, 20, 15, 11, 44),
    'level'        : 'ERROR',
    'module'       : 'ForkPoolWorker-5',
    'message'      : 'Error waiting for results: Never call result.get()...',
    'error_type'   : 'CeleryMisuse',   # classified by ERROR_PATTERNS dict
    'entity'       : 'users',          # from [ENTITY: xxx] if present
    'worker_id'    : 5,                # Celery worker number
    'db_name'      : 'bridgesec_2026-01-20T1511',
    'duration_sec' : 4.13,             # from "Time: 4.13s"
    'success_num'  : 0,                # from "Successful: 0/18"
    'success_total': 18,
}
```

`parse_file(path: str) -> list[dict]`

Runs `parse_line` on every line in the file. Returns a list of all successfully parsed dicts. Failed lines are silently skipped.

**Error classification:** The `ERROR_PATTERNS` dictionary maps regex patterns to named error types. Every parsed line's message is checked against all patterns. The first match becomes the line's `error_type`. This is how the preprocessor later counts "how many auth failures happened this minute" without re-scanning the text.

```python
ERROR_PATTERNS = {
    'OktaTokenError'  : r'Error validating Okta token',
    'Forbidden'       : r'Forbidden:',
    'InternalError'   : r'Internal Server Error',
    'OktaAccessDenied': r'Okta API access denied',
    'CeleryMisuse'    : r'Never call result\.get\(\) within a task',
    'TypeError'       : r'TypeError:',
}
```

**Why this design:** Classifying error types at parse time means the preprocessor only needs to count rows by `error_type` — no text scanning at feature-extraction time. Fast and clean.

---

### `ingestion/preprocessor.py`

**Why this file exists:** ML models cannot read text. They need numbers. The preprocessor converts a list of parsed log dicts into a numeric table — one row per minute, 13 columns.

**Why 1-minute windows:** A single log line in isolation is meaningless. One ERROR line could be a transient network blip. Thirty ERROR lines in one minute is a genuine crisis. The 1-minute window is the smallest granularity where patterns become statistically meaningful. It is also fast enough to detect incidents within one minute of them starting — acceptable for operational monitoring.

**Key function:**

`build_feature_vectors(records: list[dict], window_minutes: int = 1) -> pd.DataFrame`

Converts the list of dicts to a DataFrame, sets the timestamp as the index, and calls `.resample('1min')` to group all rows by which minute they fall into. Then computes each feature by aggregating within each group:

```python
features['error_rate']   = (df['level'] == 'ERROR').resample(freq).sum()
features['warning_rate'] = (df['level'] == 'WARNING').resample(freq).sum()
```

This counts how many ERROR lines appeared in each 1-minute bucket. The same pattern repeats for all 13 features.

**The 13 features and why each was chosen:**

| Feature | What it measures | Why it matters |
|---|---|---|
| `error_rate` | ERROR lines per minute | Spikes indicate failures |
| `warning_rate` | WARNING lines per minute | Early warning signal |
| `auth_failure_count` | Okta token validation failures | Rare in normal ops — any occurrence is suspicious |
| `forbidden_count` | HTTP 403 responses | Access control issues |
| `internal_server_error_count` | HTTP 500 responses | Application bugs or regressions |
| `okta_api_denied_count` | Okta API access denied | Permission revocation |
| `bulk_fetch_success_ratio` | successful entities / total entities | The single most important signal — 0.0 is always critical |
| `entity_completion_count` | entity processing completions | Low count means tasks aren't finishing |
| `avg_completion_time_sec` | mean time per entity task | Slowdowns indicate upstream problems |
| `celery_workers_active` | unique worker IDs seen | Worker pool health |
| `unique_error_modules` | distinct modules throwing errors | Widespread vs localised failure |
| `has_python_exception` | any TypeError present | Code-level bugs |
| `has_celery_misuse` | result.get() called inside task | Misconfiguration causing deadlocks |

**The bulk_fetch_success_ratio special case:**

This feature deserves special attention. `Successful: 0/18` in a log line means 18 entity fetch tasks ran and zero returned data. This is the most severe possible outcome. The feature is computed as `success_num / success_total` — but only when a bulk fetch actually ran. When no fetch ran in a given minute, the ratio is left as `NaN` (not zero). This distinction is critical: `NaN` means "no fetch happened" while `0.0` means "fetch happened and completely failed." Mislabelling quiet minutes as failures would corrupt all downstream models.

---

### `ingestion/log_reader.py`

**Why this file exists:** The parser and preprocessor work on completed files. The log reader enables real-time monitoring by watching a file as new lines are written to it — exactly like the Unix `tail -f` command.

**Key functions:**

`tail_file(filepath: str)`

A Python generator that opens the log file, seeks to the end, and yields new lines as they appear. Checks for new content every 100ms. Runs forever until interrupted.

`live_monitor(log_path: str, models: dict, window_seconds: int = 60)`

Maintains a rolling buffer of parsed log lines. Every `window_seconds` seconds, converts the buffer into feature vectors, runs the alert engine, prints the result, and clears the buffer. Also maintains a deque of the last 10 windows to provide the LSTM with its required sequence context.

---

## Phase 2 — Feature Labelling

### `training/label_logs.py`

**Why this file exists:** Supervised ML models need labelled training data. Since you have real logs with known problems, you can write rules based on domain knowledge to automatically label each 1-minute window.

**The labelling philosophy:** These rules encode what a senior engineer already knows about this system. They are not guesses — they are the same criteria a human would use when reviewing the logs manually.

**Key function:**

`label_features(features: pd.DataFrame) -> pd.Series`

Returns a Series of 0 (normal) or 1 (anomaly) for each row. Rules applied in order:

```python
# Only flag bulk fetch failure when a fetch actually ran
bulk = features['bulk_fetch_success_ratio']
labels[(bulk.notna()) & (bulk == 0.0)] = 1

# Auth failures — any occurrence is suspicious
labels[features['auth_failure_count'] >= 1] = 1

# Repeated internal server errors
labels[features['internal_server_error_count'] >= 3] = 1

# Okta access denied
labels[features['okta_api_denied_count'] >= 1] = 1

# Celery misuse
labels[features['has_celery_misuse'] == 1] = 1

# Python exceptions
labels[features['has_python_exception'] == 1] = 1

# Genuine error spike
labels[features['error_rate'] > 30] = 1
```

**Result on your data:** 81,968 normal windows, 103 anomaly windows (0.1% anomaly rate). This is realistic — production systems are normal most of the time.

**Why not use ML to generate labels?** You have ground truth. The architecture document describes exactly what went wrong in these logs. Writing explicit rules based on that knowledge produces more reliable labels than any unsupervised clustering approach would. Labels derived from domain knowledge are always preferable to labels derived from statistical patterns when ground truth is available.

---

## Phase 3 — The Three Models

### Why three models instead of one

Each model has a fundamentally different approach and catches different failure modes:

| Model | Training data | What it learns | Blind spot |
|---|---|---|---|
| Keras classifier | Labelled normal + anomaly | The exact boundary between normal and anomaly | Novel anomaly types never seen in training |
| TF autoencoder | Normal data only | What normal looks like | May miss subtle anomalies within the normal distribution |
| PyTorch LSTM | Sequences of windows | Whether the progression of events over time makes sense | Single-minute sudden failures with no buildup |

When two or more models agree something is wrong, confidence is much higher than any single model's verdict.

---

### `models/classifiers.py` — Keras Binary Classifier

**What it does:** Takes 13 numbers representing one minute of log activity and outputs a probability between 0 and 1 that the minute was anomalous.

**Architecture:**

```
Input (13 features)
    → Dense(64, ReLU)     # learns which feature combinations matter
    → Dropout(0.3)        # prevents overfitting
    → Dense(32, ReLU)     # refines the pattern
    → Dropout(0.3)
    → Dense(1, Sigmoid)   # outputs 0.0–1.0 probability
```

Total parameters: 3,009. Intentionally small — your classification problem is simple enough that a large model would overfit.

**Why ReLU activation:** ReLU (Rectified Linear Unit) outputs 0 for negative inputs and passes positive inputs through unchanged. This allows the network to learn non-linear patterns — for example, "auth_failures AND bulk_ratio=0 together is worse than either alone" is a non-linear relationship that ReLU layers can represent.

**Why Sigmoid on the output:** Sigmoid squashes any number into a range between 0 and 1, making it interpretable as a probability. Output of 0.97 means "97% confident this is an anomaly."

**The class imbalance problem and how SMOTE fixes it:**

With 81,968 normal rows and 103 anomaly rows, a naive model learns to always predict "normal" and achieves 99.9% accuracy while catching zero anomalies. This is useless.

SMOTE (Synthetic Minority Oversampling Technique) solves this by generating synthetic anomaly rows. It takes two real anomaly rows, computes a point somewhere between them in feature space, and creates a new synthetic anomaly at that point. After SMOTE, the training set has equal numbers of normal and anomaly examples — 81,968 of each. The model is now forced to learn what anomalies actually look like rather than taking the easy path of always predicting normal.

**Why SMOTE instead of just increasing class weight:** Class weight of 800x (the natural imbalance ratio) caused the model to predict anomaly for everything — it gamed the loss function. Class weight of 50x caused it to predict normal for everything. SMOTE eliminates the imbalance at the data level, making the model's task straightforward regardless of the weight.

**Training procedure:**

```python
# Fill NaN → scale → SMOTE → split 80/20 → train 30 epochs
X = np.nan_to_num(X, nan=0.0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X, y = SMOTE(k_neighbors=5).fit_resample(X, y)
model.fit(X_train, y_train, epochs=30, class_weight={0:1.0, 1:5.0})
```

**Why StandardScaler:** Your 13 features have wildly different ranges. `error_rate` goes from 0 to 60. `bulk_fetch_success_ratio` goes from 0.0 to 1.0. Without scaling, the model pays disproportionate attention to large-magnitude features and ignores small-magnitude ones that may be more informative. StandardScaler transforms each feature to have mean 0 and standard deviation 1, making them comparable.

**Saved files:**
- `models/classifier_model.keras` — the trained network weights
- `models/scaler.pkl` — the fitted StandardScaler (must always be used together with the model)

**Results:** 100% recall (zero missed anomalies), 99.9% precision (8 false alarms out of 16,394 normal test windows).

---

### `models/autoencoder.py` — TensorFlow Autoencoder

**What it does:** Learns what normal log windows look like by training exclusively on normal data. At inference time, feeds any window through the network and measures how well it can reconstruct it. Normal windows reconstruct well (low error). Anomalous windows reconstruct poorly (high error). Reconstruction error is the anomaly score.

**Why this is fundamentally different from the classifier:** The classifier was told "here are anomaly examples — learn to recognise them." The autoencoder is never shown anomaly examples. It only sees normal data. It learns the structure of normality and then flags anything that doesn't fit that structure. This means it can catch completely new types of failures that never appeared in training data — something the classifier cannot do.

**Architecture:**

```
Input (13 features)
    → Dense(8, ReLU)     # encoder: compress to 8
    → Dense(4, ReLU)     # encoder: compress to 4 (bottleneck)
    → Dense(8, ReLU)     # decoder: expand back to 8
    → Dense(13, linear)  # decoder: reconstruct original 13 features
```

The network is trained with `input = output` — it tries to reproduce its own input. The bottleneck (4 neurons) forces it to learn a compressed representation of normal patterns. It cannot memorise every input because there is not enough capacity in the bottleneck.

**Why MSE loss:** Mean Squared Error measures how far the reconstructed values are from the original values. The model minimises this during training on normal data. After training, a normal window that looks like the training distribution will have very low MSE. An anomaly window that looks nothing like the training distribution will have very high MSE.

**The threshold:** After training, reconstruction errors are computed for all windows. The threshold is set at the 99th percentile of normal window errors. Any window with reconstruction error above this threshold is flagged as anomalous. Using 99th percentile (rather than 95th) reduces false alarms while still catching genuine anomalies, which have reconstruction errors orders of magnitude higher than normal windows.

**Why 99th percentile specifically:** Your normal windows have reconstruction errors averaging 0.002. Your anomaly windows average 409 — a 200,000x difference. The threshold only needs to be above the noisy tail of the normal distribution to perfectly separate the two classes. Setting it at the 99th percentile of normal errors leaves a comfortable margin above normal noise while remaining far below anomaly territory.

**Results on your data:**
```
Normal windows   — mean reconstruction error: 0.0019
Anomaly windows  — mean reconstruction error: 409.69
Threshold                                   : 0.0091
```

This gap is enormous. The model has clearly learned what normal looks like and is completely unable to reconstruct anomaly patterns.

**Saved files:**
- `models/autoencoder_model.keras`
- `models/autoencoder_scaler.pkl`
- `models/autoencoder_threshold.pkl` — the computed 99th percentile threshold

---

### `models/lstm_detector.py` — PyTorch LSTM

**What it does:** Reads a sequence of 10 consecutive 1-minute windows and predicts whether an anomaly occurred anywhere in that sequence. Unlike the classifier and autoencoder which look at one minute in isolation, the LSTM has memory — it can detect patterns that develop over multiple minutes.

**What LSTM stands for:** Long Short-Term Memory. A type of recurrent neural network that maintains a hidden state (memory) as it reads through a sequence. At each step, the current input and the previous hidden state together determine the output. This allows the network to relate what happened 5 minutes ago to what is happening now.

**Architecture:**

```python
class LSTMDetector(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32, num_layers=2):
        self.lstm = nn.LSTM(input_size=13, hidden_size=32, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            # no sigmoid here — BCEWithLogitsLoss applies it internally
        )
```

The LSTM processes 10 time steps (10 minutes). After the final step, the last hidden state is fed through a small classifier network that outputs a single anomaly score.

**Why PyTorch instead of Keras for this model:** PyTorch requires you to write the training loop explicitly. This is intentional for learning purposes — you see every step:

```python
for epoch in range(20):
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)     # 1. forward pass
        loss   = criterion(y_pred, y_batch)  # 2. compute loss
        optimizer.zero_grad()        # 3. clear previous gradients
        loss.backward()              # 4. backpropagate
        optimizer.step()             # 5. update weights
```

This is the fundamental loop that every neural network training algorithm in existence implements. Keras hides it. PyTorch shows it.

**Why BCEWithLogitsLoss instead of BCELoss:** `BCEWithLogitsLoss` combines the sigmoid activation and binary cross-entropy loss into one numerically stable operation. Using `BCELoss` with a Sigmoid output layer applies sigmoid twice — once in the layer, once in the loss — which destroys the gradients and prevents learning. The model output is raw logits; sigmoid is applied manually only during inference.

**Sequence construction:** The `build_sequences` function slides a window of length 10 across all 82,061 windows. Each sequence is labelled 1 if any window within it was anomalous. This means the model learns to flag sequences that contain or lead up to anomalies — not just the single anomalous minute itself.

**Why the test split must be stratified:** Your 103 anomaly windows are clustered at specific timestamps. A time-ordered 80/20 split puts all anomalies in the training set and none in the test set. Stratified split ensures anomalies appear in both — 570 anomaly sequences in training, 143 in test.

**Results:** 99.3% recall (142/143 anomaly sequences caught), 93% precision (10 false alarms out of 16,270 normal sequences).

**Why LSTM scores 0.0 on sudden single-minute failures:** The LSTM reads the 10 minutes preceding the anomaly. If those minutes were completely normal, the LSTM correctly reports nothing suspicious is building up. The LSTM's strength is detecting gradual degradation — rising error rates, slowly declining success ratios — not single-minute sudden failures. That is why all three models are needed.

**Saved files:**
- `models/lstm_model.pt` — PyTorch state dict (model weights only)
- `models/lstm_scaler.pkl` — fitted StandardScaler

---

## Phase 4 — Alert Engine

### `alert/alert_engine.py`

**What it does:** Loads all three trained models, runs them on each new feature window, combines their outputs through an ensemble voting system, and applies hard-coded rule overrides for the most critical failure patterns.

**Why an ensemble:** Each model has different blind spots. The classifier catches known patterns but misses novel failures. The autoencoder catches novel patterns but generates more false alarms. The LSTM catches temporal patterns but misses sudden single-minute events. When two or more models agree, the combined confidence is much higher than any individual model's confidence. False alarms from one model are suppressed when the other two say normal.

**Hard rules — why they exist and what they are:**

Some conditions are so definitively serious that ML voting is unnecessary and potentially dangerous. If bulk fetch returned 0 successes, you don't need three models to agree — that is always CRITICAL. Hard rules fire before ML models are consulted:

```python
if bulk == 0.0 and bulk is not NaN:  → CRITICAL
if auth_failure_count >= 2:          → CRITICAL
if internal_server_error_count >= 3: → HIGH
if okta_api_denied_count >= 3:       → HIGH
```

**Alert severity levels:**

| Severity | Condition | Meaning | Recommended action |
|---|---|---|---|
| NORMAL | 0 votes, no hard rule | All models agree this minute is fine | Log and ignore |
| LOW | 1 vote | One model is uncertain | Log for review, do not alert |
| HIGH | 2 votes OR hard rule (500s/Okta) | Strong evidence of a problem | Send alert to on-call |
| CRITICAL | 3 votes OR hard rule (bulk/auth) | Overwhelming evidence | Immediate response required |

**Key functions:**

`load_models() -> dict`

Loads all three models and their associated scalers/thresholds from disk. Called once at startup. Returns a dictionary containing all loaded objects.

`check_hard_rules(feature_row: dict) -> tuple | None`

Checks the feature values against hard-coded thresholds. Returns (severity, reason) if any rule fires, otherwise None.

`run_models(feature_row, models, lstm_sequence) -> dict`

Runs the classifier, autoencoder, and LSTM on the current window. Each model produces a vote (0 or 1). Returns all scores and the total vote count.

`decide_alert(votes, hard_rule) -> tuple`

Maps vote count and hard rule presence to a final severity level and reason string.

`analyze_window(feature_row, feature_dict, models, lstm_sequence) -> dict`

The main entry point. Fills NaN values, checks hard rules, runs models, decides severity. Returns a complete result dictionary:

```python
{
    'severity' : 'CRITICAL',
    'reason'   : 'Bulk fetch returned 0 successes',
    'votes'    : 2,
    'hard_rule': True,
    'scores'   : {
        'classifier' : {'score': 1.000, 'vote': 1},
        'autoencoder': {'score': 200.1, 'vote': 1},
        'lstm'       : {'score': 0.000, 'vote': 0},
        'total_votes': 2
    }
}
```

---

## Understanding the Output

### What each alert means in plain English

**NORMAL — 0/3 votes**
```
[15:12:00] NORMAL | votes=0/3 | All models agree: normal window
```
All three models looked at this minute and found nothing unusual. The minute had the error rates, feature counts, and temporal patterns consistent with what the system normally produces. No action needed.

**LOW — 1/3 votes**
```
[15:23:00] LOW | votes=1/3 | One model flagged this window
           clf=0.12 | ae=0.015 | lstm=0.02
```
One model found something slightly unusual. This often happens to the autoencoder on minutes immediately following a real failure — the system is recovering and activity is slightly above baseline. The other two models say normal, which suppresses the alert. Log it for review but do not page anyone.

**HIGH — 2/3 votes**
```
[19:17:00] HIGH [!!] | votes=2/3 | Two models agree: anomaly detected
           clf=1.000 | ae=203.1 | lstm=0.000
```
Two independent models both flagged this minute as anomalous. The classifier is 100% confident. The autoencoder reconstruction error is 203 — massively above its 0.009 threshold. Only the LSTM disagrees, which makes sense if the anomaly was sudden (no temporal buildup). This warrants an immediate alert to the on-call engineer.

**CRITICAL — hard rule fired**
```
[15:37:00] CRITICAL [!!!] | votes=2/3 | Bulk fetch returned 0 successes
           clf=1.000 | ae=200.1 | lstm=0.000
```
A hard rule fired before ML voting even ran. `bulk_fetch_success_ratio = 0.0` means 18 entity fetch tasks ran and zero returned data. This is a complete system failure. The ML scores confirm it (classifier=1.0, autoencoder error=200). Wake someone up immediately.

### Reading the model scores

| Score | Model | Interpretation |
|---|---|---|
| `clf=1.000` | Classifier | 100% confident this is an anomaly |
| `clf=0.050` | Classifier | 5% confident — treating as normal |
| `ae=200.1` | Autoencoder | Reconstruction error 200 — 22,000x above normal mean of 0.009 |
| `ae=0.002` | Autoencoder | Reconstruction error near-zero — looks perfectly normal |
| `lstm=0.950` | LSTM | 95% confident the sequence leading to this window is anomalous |
| `lstm=0.000` | LSTM | No anomalous temporal pattern detected |

---

## Live Monitoring

### Running against a live log file

```bash
python main.py /path/to/your/app.log
```

The system tails the file in real time. Every 60 seconds it processes all new lines that appeared in that window, builds feature vectors, runs all three models, and prints the alert level. If your Django or Celery process is writing to this file right now, the system is monitoring it right now.

### Demo mode — replaying app.log as if it's live

Open two terminals side by side.

Terminal A:
```bash
python demo_replayer.py
```

Terminal B:
```bash
python main.py data/reference/demo_live.log
```

The replayer writes lines from `app.log` into a new file at controlled speed. The monitor reads them as they appear. When the replayer reaches the 15:11 bulk fetch failure, the monitor fires a CRITICAL alert in real time. This demonstrates the full pipeline end to end with visible, real-time output.

---

## Model Performance Summary

| Model | Recall | Precision | False alarms (test) | Missed anomalies |
|---|---|---|---|---|
| Keras classifier | 100% | 99.9% | 8 | 0 |
| TF autoencoder | 100% | 13% | 712 | 0 |
| PyTorch LSTM | 99.3% | 93% | 10 | 1 |
| Ensemble (HIGH+CRITICAL) | ~99% | ~99% | near-zero | near-zero |

The autoencoder's 13% precision (712 false alarms) looks bad in isolation but is acceptable in the ensemble — those false alarms are filtered out when the other two models say normal. Only when the autoencoder and at least one other model both flag a window does a HIGH alert fire.

---

## Design Decisions

**Why 1-minute windows:** Fast enough to catch incidents within one minute of onset. Large enough that each bucket contains enough lines to compute meaningful statistics. Your busiest minute had 478 log lines — sufficient signal for all 13 features.

**Why three different frameworks:** Keras for the classifier (cleanest API, fastest to build), TensorFlow for the autoencoder (natural fit for encoder/decoder architecture in the Keras functional API), PyTorch for the LSTM (explicit training loop makes the learning algorithm transparent, most widely used in ML research).

**Why SMOTE over class weights:** Class weight of 800x (the natural imbalance ratio) causes the model to predict anomaly for everything — it minimises loss by paying the 1-point penalty 82,000 times rather than risking the 800-point penalty even once. SMOTE eliminates the imbalance problem at the data level, making training stable and predictable.

**Why the autoencoder trains on normal data only:** If anomaly examples were included in training, the autoencoder would learn to reconstruct them too — eliminating its ability to detect them via reconstruction error. Training only on normal data preserves the property that reconstruction error is a reliable anomaly signal.

**Why the LSTM uses stratified split:** The 103 real anomaly windows are temporally clustered. Time-ordered splits put all anomalies in training and none in test, making evaluation impossible. Stratified split guarantees the anomaly proportion is preserved in both sets.

**Why hard rules exist alongside ML:** ML models can be uncertain near decision boundaries. A bulk fetch returning 0/18 successes is unambiguously a complete system failure — no uncertainty, no boundary case. Hard rules handle these clear-cut situations instantly and reliably without depending on model confidence.

---

## Requirements

```
torch>=2.0
tensorflow>=2.13
scikit-learn
imbalanced-learn
pandas
numpy
python-dateutil
```

---

## Extending the System

**Retraining:** As your system accumulates more log data, retrain all three models monthly. New anomaly patterns will be captured in the training data and the models will learn them.

**Adding alert notifications:** Implement `alert/notifier.py` to send Slack messages, emails, or PagerDuty alerts when severity reaches HIGH or CRITICAL. The `analyze_window` return dict contains all the information needed to compose a useful alert message.

**Adding new features:** Add new entries to `FEATURE_COLS` in `preprocessor.py` and a corresponding computation in `build_feature_vectors`. Retrain all three models after adding features.

**Connecting to ELK or Splunk:** The parsed log dicts and feature vectors can be shipped to Elasticsearch or Splunk's HTTP Event Collector. The architecture includes `integrations/elk_forwarder.py` and `integrations/splunk_hec.py` stubs for this purpose.
