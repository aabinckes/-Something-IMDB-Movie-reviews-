#loads dataset IMDb from Hugging Face

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

DATASET_NAME = "stanfordnlp/imdb"

def clean_text(s: str) -> str:
    return s.replace("<br />", " ").strip()

@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.2
    seed: int = 42
    splits_dir: str = "splits"
    clean: bool = True
    config_name: Optional[str] = None

def load_imbd(cache_dir: Optional[str] = None, config_name: Optional[str] = None) -> DatasetDict:
    if config_name is None:
        return load_dataset(DATASET_NAME, cache_dir=cache_dir)
    return load_dataset(DATASET_NAME, config_name, cache_dir=cache_dir)

def get_splits(cache_dir: Optional[str] = None, cfg: SplitConfig = SplitConfig()) -> DatasetDict:
    """
    Returns DatasetDict with train, val, test
    -train/val are created by splitting the official train split
    -split indices are saved/loaded from cfg.splits_dir so it is reproducible
    """
    ds = load_imbd(cache_dir=cache_dir, config_name=cfg.config_name)

    train_full = ds["train"]
    test = ds["test"]
    unsup = ds["unsupervised"] if "unsupervised" in ds else None

    splits_dir = Path(cfg.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"imdb_train_val_seed{cfg.seed}_val{cfg.val_size}.json"

    if split_path.exists():
        split_obj = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = split_obj["train_idx"]
        val_idx = split_obj["val_idx"]
    else:
        indices = list(range(len(train_full)))
        labels = train_full["label"] #0/1
        train_idx, val_idx = train_test_split(
            indices,
            test_size=cfg.val_size,
            random_state=cfg.seed,
            stratify=labels,
        )

        split_obj: Dict[str, Any] = {
            "dataset": DATASET_NAME,
            "config_name": cfg.config_name,
            "val_size": cfg.val_size,
            "seed": cfg.seed,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "train_idx": train_idx,
            "val_idx": val_idx,
        }
        split_path.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
    
    train = train_full.select(train_idx)
    val = train_full.select(val_idx)

    if cfg.clean:
        train = train.map(lambda ex: {"text": clean_text(ex["text"])})
        val = val.map(lambda ex: {"text": clean_text(ex["text"])})
        test = test.map(lambda ex: {"text": clean_text(ex["text"])})
        if unsup is not None:
            unsup = unsup.map(lambda ex: {"text": clean_text(ex["text"])})
        
    out = DatasetDict({"train": train, "val": val, "test": test})
    if unsup is not None:
        out["unsupervised"] = unsup
    
    return out