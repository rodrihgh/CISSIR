"""
Hack for local module imports: https://stackoverflow.com/a/51028921

Must be imported before any other local cissir module in any notebook.
"""

import sys
from pathlib import Path

module_path = Path(__file__).parents[1].absolute()
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
