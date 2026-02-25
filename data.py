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

