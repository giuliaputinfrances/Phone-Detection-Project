#!/usr/bin/env python
"""Convenience shim: `python scripts/check_env.py` == `pdp check-env`."""

import sys

from pdp.env import check

if __name__ == "__main__":
    raise SystemExit(check(allow_cpu="--allow-cpu" in sys.argv))
