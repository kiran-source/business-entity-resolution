"""
Root CLI entry point for Business Entity Resolution System.
"""
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(CURRENT_DIR, "business_entity_resolution")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from business_entity_resolution.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
