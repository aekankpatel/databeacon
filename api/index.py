"""Vercel entrypoint: the DataBeacon FastAPI app as a Python function.

Vercel routes /api/* here (see vercel.json). The backend package lives in backend/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from databeacon.api.main import app  # noqa: E402,F401
