#logistic regression training file
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data import get_splits, SplitConfig

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

def main() -> None:
    p = argparse.ArgumentParser()

    #data/split
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config_name", type=str, default=None)
    p.add_argument("--no_clean", action="store_true")

    #TF-IDF
    p.add_argument("--max_features", type=int, default=50000)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)

    #logistic Regression
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class_weight", type=str, default=None, choices=[None, "balanced"])
    p.add_argument("--max_iter", type=int, default=2000)

    #output
    p.add_argument("--out_dir", type=str, default="results/logreg_baseline")

    args = p.parse_args()

    cfg = SplitConfig(
        val_size=args.val_size,
        seed=args.seed,
        splits_dir=args.splits_dir,
        clean=(not args.no_clean),
        config_name=args.config_name,
    )

    ds = get_splits(cache_dir=args.cache_dir, cfg=cfg)
    train_ds = ds["train"]
    val_ds = ds["val"]

    X_train_text = train_ds["text"]
    y_train = np.array(train_ds["label"], dtype=int)

    X_val_text = val_ds["text"]
    y_val = np.array(val_ds["label"], dtype=int)

    #TF-IDF only on train
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    model = LogisticRegression(
        C=args.C,
        solver="liblinear",
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    metrics = evaluate(y_val, val_pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "tfidf.joblib")
    joblib.dump(model, out_dir / "logreg.joblib")

    run_info = {
        "dataset": "stanfordnlp/imdb",
        "split": {
            "val_size": args.val_size,
            "seed": args.seed,
            "splits_dir": args.splits_dir,
        },
        "tfidf": {
            "max_features": args.max_features,
            "ngram_range": [args.ngram_min, args.ngram_max]
            "min_df": args.min_df,
            "max_df": args.max_df,
        },
        "logreg": {
            "C": args.C,
            "solver": "liblinear",
            "class_weight": args.class_weight,
            "max_iter": args.max_iter,
        },
        "val_metrics": metrics,
    }

    (out_dir / "metrics.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print("TF-IDF shapes:", X_train.shape, "(train)", X_val.shape, "(val)")
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()