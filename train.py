"""One-shot training runner — run this to train all models."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_models import train_all

if __name__ == "__main__":
    train_all(test_fraction=0.15, n_iter=40)
