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