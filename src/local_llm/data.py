"""Training text: download once, then read from disk."""

import urllib.request
from pathlib import Path

DATA_DIR = Path("data.nosync")
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_tiny_shakespeare(data_dir: Path = DATA_DIR) -> str:
    """Return ~1 MB of Shakespeare plays as one string (downloads on first use)."""
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")
