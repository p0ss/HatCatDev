"""Allow running as: python -m chorion"""
import sys
from pathlib import Path

# Ensure paths are set up before any chorion imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from chorion.server import main

main()
