#!/usr/bin/env python3
"""Run the asynchronous real-time annotation path on a recorded episode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevfabric.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
