#!/usr/bin/env python3
"""
Script to run Lee2019 SSVEP debugging using the saturation detector.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.saturation_detector import debug_main

if __name__ == "__main__":
    print("Running Lee2019 SSVEP debugging...")
    debug_results = debug_main()
    print("Debugging completed!")
