# Repo-root conftest: puts the shared harness on sys.path for every project's tests.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "common" / "py"))
