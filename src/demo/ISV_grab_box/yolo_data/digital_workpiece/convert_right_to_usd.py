#!/usr/bin/env python3
"""Convert the LR028550 right-side trim using the paired LR028551 material standard."""

from pathlib import Path

import convert_left_to_usd as converter


converter.SOURCE = Path(__file__).with_name("right.stp").resolve()
converter.OUTPUT = Path(__file__).with_name("right.usd").resolve()
converter.PART_NUMBER = "Land Rover LR028550"


if __name__ == "__main__":
    converter.main()
