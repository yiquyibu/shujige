"""Backward-compatible entry point for hybrid_search in 检索系统."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from hybrid_search import main

if __name__ == "__main__":
    main()
