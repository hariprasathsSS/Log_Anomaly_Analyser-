import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def build_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation='`relu`')(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_classifier(features, labels):
    X = features.values.astype(np.float32)
    y = labels.values.astype(np.float32)

    # Fill NaN before scaling and SMOTE
    # bulk_fetch_success_ratio NaN means no fetch ran → treat as 1.0 (perfect, nothing failed)
    X = np.nan_to_num(X, nan=0.0)
    # Scale all features to same range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    

    print(f"Before SMOTE: normal={( y==0).sum()}, anomaly={(y==1).sum()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X, y = sm.fit_resample(X, y)
    print(f"After SMOTE : normal={(y==0).sum()}, anomaly={(y==1).sum()}")

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Much gentler class weight — 50x instead of 800x
    # 800x made the model call everything anomaly
    class_weight = {0: 1.0, 1: 5.0}
    print(f"Class weight for anomaly: {class_weight[1]}x")

    model = build_model(X_train.shape[1])
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=512,
        validation_split=0.1,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate using sklearn — more reliable than keras metrics
    print("\nEvaluating...")
    y_pred_prob = model.predict(X_test, verbose=0)

    # Try different thresholds — default 0.5 may miss anomalies
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        print(f"\nThreshold = {threshold}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred,
              target_names=['normal', 'anomaly'], zero_division=0))

    # Save with best threshold (0.3 is usually best for anomaly detection)
    model.save('models/classifier_model.keras')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\nSaved → models/classifier_model.keras")

    return model, scaler, history




# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pickle

# def build_model(input_dim: int) -> keras.Model:
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy',
#                  keras.metrics.Precision(name='precision'),
#                  keras.metrics.Recall(name='recall')]
#     )
#     return model

# def train_classifier(features, labels):
#     X = features.values.astype(np.float32)
#     y = labels.values.astype(np.float32)

#     # Scale — bring all 13 features to same range
#     # without this, error_rate=31 dominates bulk_ratio=0.0
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     # Split — 80% train, 20% test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Class weight — fixes the 0.1% anomaly problem
#     # tells keras: missing an anomaly costs 800x more than missing a normal
#     total    = len(y_train)
#     n_normal  = (y_train == 0).sum()
#     n_anomaly = (y_train == 1).sum()
#     class_weight = {
#         0: 1.0,
#         1: n_normal / n_anomaly
#     }
#     print(f"Class weight for anomaly: {class_weight[1]:.1f}x")

#     model = build_model(X_train.shape[1])

#     print("\nTraining...")
#     history = model.fit(
#         X_train, y_train,
#         epochs=20,
#         batch_size=256,
#         validation_split=0.1,
#         class_weight=class_weight,
#         verbose=1
#     )

#     # Evaluate on test set
#     print("\nTest set results:")
#     results = model.evaluate(X_test, y_test, verbose=0)
#     metrics = dict(zip(model.metrics_names, results))
#     print(f"  Accuracy  : {metrics['accuracy']*100:.1f}%")
#     print(f"  Precision : {metrics['precision']*100:.1f}%")
#     print(f"  Recall    : {metrics['recall']*100:.1f}%")

#     # What precision and recall mean here:
#     # Precision: of all windows we flagged as anomaly, how many were actually anomaly?
#     # Recall: of all actual anomalies, how many did we catch?

#     # Save model and scaler
#     model.save('models/classifier_model.keras')
#     with open('models/scaler.pkl', 'wb') as f:
#         pickle.dump(scaler, f)
#     print("\nSaved → models/classifier_model.keras")

#     return model, scaler, history