"""
Baseline intent classifier: TF-IDF + Logistic Regression.

This is the "simple baseline" for comparison in the eval report -- not the final model. 
Trains on labeled golden-set batches (small data, so we use cross-validation instead of a held-out split for 
training-fit sanity checks; a genuinely held-out chunk of the golden set should still be reserved for
final reported eval numbers, not reused here).
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "data" / "golden_set"


def load_labeled_data(golden_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(golden_dir.glob("labeling_batch_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        df = df[df["intent"].notna()]
        df = df[df["intent"].astype(str).str.strip() != ""]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No labeled batches found in {golden_dir}")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def main(args):
    data = load_labeled_data(GOLDEN_DIR)
    print(f"Loaded {len(data)} labeled examples across "
          f"{data['intent'].nunique()} intents")
    print(data["intent"].value_counts())

    X = data["customer_text"].tolist()
    y = data["intent"].tolist()

    n_splits = min(5, data["intent"].value_counts().min())
    if n_splits < 2:
        print("WARNING: smallest intent class has <2 examples — "
              "cross-validation results will be unreliable until more data is labeled.")
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_preds = cross_val_predict(dummy, X, y, cv=cv)
    print("\n=== TRIVIAL BASELINE (majority class) ===")
    print(classification_report(y, dummy_preds, zero_division=0))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X_vec = vectorizer.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    preds = cross_val_predict(clf, X_vec, y, cv=cv)

    print("\n=== SIMPLE BASELINE (TF-IDF + Logistic Regression) ===")
    print(classification_report(y, preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    clf.fit(X_vec, y)
    import joblib
    model_dir = REPO_ROOT / "src" / "classifier" / "artifacts"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf}, model_dir / "tfidf_logreg.joblib")
    print(f"\nSaved fitted model to {model_dir / 'tfidf_logreg.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())