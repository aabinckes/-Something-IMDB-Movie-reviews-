import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
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
    # Standard Data Args
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=10000)
    
    # Neural Network Hyperparameters
    p.add_argument("--hidden_layers", type=int, nargs='+', default=[100])
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.0001)
    p.add_argument("--out_dir", type=str, default="results/nn_baseline")

    args = p.parse_args()
    cfg = SplitConfig(val_size=args.val_size, seed=args.seed)
    ds = get_splits(cfg=cfg)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train = vectorizer.fit_transform(ds["train"]["text"])
    X_val = vectorizer.transform(ds["val"]["text"])
    y_train = np.array(ds["train"]["label"])
    y_val = np.array(ds["val"]["label"])

    print(f"Training Neural Network with layers {args.hidden_layers}...")
    
    # MLP = Multi-Layer Perceptron (Feedforward NN)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        max_iter=args.max_iter,
        alpha=args.alpha,
        random_state=args.seed,
        verbose=True
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    val_pred = model.predict(X_val)
    metrics = evaluate(y_val, val_pred)

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "nn_model.joblib")
    
    results = {"params": vars(args), "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    
    print(f"\nValidation Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()