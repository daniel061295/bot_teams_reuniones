import sys
from pathlib import Path

# Add root to the path for correct imports during testing
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
