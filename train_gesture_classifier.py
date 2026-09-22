"""
train_gesture_classifier.py
===========================================================
Trains a gesture classifier from MyVoice landmark data, and exports it
to JSON so it can run *inside the browser* — closing the loop between
the Python model and the live app.

PIPELINE
  1. Load CSV(s): each row = label + 21 hand landmarks (x,y) = 42 numbers.
  2. Normalise the landmarks (see below) — the key feature-engineering step.
  3. Train K-Nearest Neighbours and Random Forest.
  4. Evaluate: accuracy, confusion matrix, per-class report, feature importance.
  5. Export the better model to gesture_model.json for the browser.

WHY NORMALISE?
  Raw landmark coordinates encode WHERE the hand was in the frame and HOW
  BIG it appeared. Two identical peace signs at different distances or
  screen positions become very different feature vectors, so a model
  trained on raw coordinates partly memorises the recording setup instead
  of the gesture shape. We fix that by:
    - translating so the wrist sits at (0,0)   -> position-invariant
    - scaling by the hand's own span           -> size/distance-invariant
  The browser applies the identical transform before predicting, so
  training and inference always agree.

USAGE
    pip3 install pandas scikit-learn joblib

    # one dataset (auto train/test split)
    python3 train_gesture_classifier.py gesture_dataset.csv

    # recommended: train on one session, test on a SEPARATE session
    python3 train_gesture_classifier.py session1.csv session2.csv

    # also report raw-vs-normalised accuracy (good for a write-up)
    python3 train_gesture_classifier.py session1.csv session2.csv --compare
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


# ---------------------------------------------------------------- features
def normalise_landmarks(X):
    """Translate to wrist origin, then scale by hand span.

    X: array-like (n_samples, 42) ordered x0,y0,x1,y1,...,x20,y20.
    Mirrors normaliseLandmarks() in MyVoice.html exactly — change both together.
    """
    X = np.asarray(X, dtype=float)
    pts = X.reshape(len(X), 21, 2)
    pts = pts - pts[:, 0:1, :]                       # wrist -> origin
    spans = np.sqrt((pts ** 2).sum(axis=2)).max(axis=1)
    spans[spans == 0] = 1.0
    pts = pts / spans[:, None, None]                 # scale to unit span
    return pts.reshape(len(X), 42)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {csv_path}.")
    y = df["label"]
    X = df.drop(columns=["label"])
    print(f"Loaded {len(df)} samples from {csv_path} — {y.nunique()} classes:")
    print(y.value_counts().to_string())
    print()
    return X, y


# ---------------------------------------------------------------- evaluate
def evaluate(model, name, X_test, y_test, quiet=False):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    if quiet:
        return acc
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.2%}")
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print()
    print(classification_report(y_test, pred, zero_division=0))
    return acc


def print_feature_importance(rf, feature_names, top_n=8):
    imp = pd.Series(rf.feature_importances_, index=feature_names)
    grouped = {}
    for feat, score in imp.items():
        idx = int(feat[1:])
        name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else feat
        grouped[name] = grouped.get(name, 0) + score
    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
    print("--- Feature importance (Random Forest) ---")
    print("Landmarks the model relied on most:")
    for name, score in ranked[:top_n]:
        print(f"  {name:<14s} {score:.3f}  {'#' * max(1, int(score * 120))}")
    print()


# ---------------------------------------------------------------- export
def tree_to_dict(tree, classes):
    t = tree.tree_

    def node(i):
        if t.children_left[i] == -1:
            return {"leaf": str(classes[int(np.argmax(t.value[i][0]))])}
        return {
            "f": int(t.feature[i]),
            "t": round(float(t.threshold[i]), 6),
            "l": node(int(t.children_left[i])),
            "r": node(int(t.children_right[i])),
        }

    return node(0)


def export_json(model, kind, classes, path):
    if kind == "knn":
        payload = {
            "type": "knn",
            "normalised": True,
            "k": int(model.n_neighbors),
            "classes": [str(c) for c in classes],
            "X": [[round(float(v), 5) for v in row] for row in model._fit_X],
            "y": [str(model.classes_[i]) for i in model._y],
        }
    else:
        payload = {
            "type": "random_forest",
            "normalised": True,
            "classes": [str(c) for c in classes],
            "trees": [tree_to_dict(est, list(model.classes_)) for est in model.estimators_],
        }

    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"Exported {kind} -> {path} ({len(json.dumps(payload)) / 1024:.0f} KB)")
    print("Load it in the app:  Gestures tab -> 'Use trained model'")


# ---------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if not args:
        print("Usage:")
        print("  python3 train_gesture_classifier.py <dataset.csv>")
        print("  python3 train_gesture_classifier.py <train.csv> <test.csv>")
        print("  (add --compare to also report raw-coordinate accuracy)")
        sys.exit(1)

    train_path = args[0]
    test_path = args[1] if len(args) > 1 else None

    X_full, y_full = load_dataset(train_path)
    feature_names = list(X_full.columns)

    smallest = y_full.value_counts().min()
    if smallest < 8:
        print(f"WARNING: smallest class has only {smallest} samples. "
              "Aim for 30-50 per gesture.\n")

    if test_path:
        print("Testing on a SEPARATE session — the honest measure of "
              "generalisation, not same-session memorisation.\n")
        X_train_raw, y_train = X_full, y_full
        X_test_raw, y_test = load_dataset(test_path)
    else:
        print("No separate test file given — splitting one dataset. NOTE: both "
              "halves come from the same session, so accuracy will look better "
              "than real-world performance. Collect a second session and pass "
              "it as a second argument for an honest number.\n")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.25, random_state=42, stratify=y_full)

    if "--compare" in flags:
        print("=" * 58)
        print("FEATURE COMPARISON: raw coordinates vs normalised")
        print("=" * 58)
        r1 = evaluate(KNeighborsClassifier(n_neighbors=5).fit(X_train_raw, y_train),
                      "", X_test_raw, y_test, quiet=True)
        r2 = evaluate(RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_raw, y_train),
                      "", X_test_raw, y_test, quiet=True)
        n1 = evaluate(KNeighborsClassifier(n_neighbors=5).fit(normalise_landmarks(X_train_raw), y_train),
                      "", normalise_landmarks(X_test_raw), y_test, quiet=True)
        n2 = evaluate(RandomForestClassifier(n_estimators=200, random_state=42).fit(normalise_landmarks(X_train_raw), y_train),
                      "", normalise_landmarks(X_test_raw), y_test, quiet=True)
        print(f"  raw coordinates -> KNN {r1:.2%} | RandomForest {r2:.2%}")
        print(f"  normalised      -> KNN {n1:.2%} | RandomForest {n2:.2%}")
        print("\n(The difference IS your feature-engineering result — report it.)\n")

    X_train = normalise_landmarks(X_train_raw)
    X_test = normalise_landmarks(X_test_raw)

    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

    knn_acc = evaluate(knn, "K-Nearest Neighbours (normalised)", X_test, y_test)
    rf_acc = evaluate(rf, "Random Forest (normalised)", X_test, y_test)
    print_feature_importance(rf, feature_names)

    if rf_acc >= knn_acc:
        best, kind, name, acc = rf, "random_forest", "Random Forest", rf_acc
    else:
        best, kind, name, acc = knn, "knn", "K-Nearest Neighbours", knn_acc

    joblib.dump(best, "gesture_model.joblib")
    print(f"Best model: {name} ({acc:.2%}) -> gesture_model.joblib")
    export_json(best, kind, sorted(pd.unique(y_train)), "gesture_model.json")


if __name__ == "__main__":
    main()
