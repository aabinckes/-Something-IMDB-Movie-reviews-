from sklearn.ensemble import RandomForestClassifier
from data import get_splits, SplitConfig
import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

def main() -> None:
    p = argparse.ArgumentParser()
    # Standard Data Args
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=10000)
    
    # Random Forest Hyperparameters
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="results/rf_baseline")

    args = p.parse_args()
    cfg = SplitConfig(val_size=args.val_size, seed=args.seed)
    ds = get_splits(cfg=cfg)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train = vectorizer.fit_transform(ds["train"]["text"])
    X_val = vectorizer.transform(ds["val"]["text"])
    y_train = np.array(ds["train"]["label"])
    y_val = np.array(ds["val"]["label"])


    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed)
    clf.fit(X_train, y_train)


    # Evaluate
    val_pred = clf.predict(X_val)
    metrics = evaluate(y_val, val_pred)

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "rf_model.joblib")
    
    results = {"params": vars(args), "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    
    print(f"\nValidation Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()