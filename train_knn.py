import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data import get_splits, SplitConfig

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
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=5000) # Reduced for speed
    
    # kNN specific arguments
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--metric", type=str, default="cosine") 
    p.add_argument("--out_dir", type=str, default="results/knn_baseline")

    args = p.parse_args()
    cfg = SplitConfig(val_size=args.val_size, seed=args.seed)
    ds = get_splits(cfg=cfg)

    # Prepare data
    X_train_text = ds["train"]["text"]
    y_train = np.array(ds["train"]["label"])
    X_val_text = ds["val"]["text"]
    y_val = np.array(ds["val"]["label"])

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    # Train kNN
    print(f"Training kNN with k={args.k} and metric={args.metric}...")
    model = KNeighborsClassifier(n_neighbors=args.k, metric=args.metric)
    model.fit(X_train, y_train)

    # Evaluate
    val_pred = model.predict(X_val)
    metrics = evaluate(y_val, val_pred)

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "knn_model.joblib")
    
    results = {"params": vars(args), "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()