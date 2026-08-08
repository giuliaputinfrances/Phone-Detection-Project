#!/usr/bin/env python
"""Convenience shim: `python scripts/list_cameras.py` == `pdp cameras`."""

from pdp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["cameras"]))
