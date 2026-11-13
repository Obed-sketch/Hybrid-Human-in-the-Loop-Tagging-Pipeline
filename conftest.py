# Make the repo-root modules (train_classifier, tag, review_queue)
# importable from the tests/ directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
