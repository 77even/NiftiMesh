#!/usr/bin/env python3
"""Minimal end-to-end example: a NIfTI segmentation -> per-label STL files.

    python examples/convert.py            # uses the bundled lung-lobe sample
    python examples/convert.py my_seg.nii.gz out_dir csg
"""

import sys
from pathlib import Path

from niftimesh import nifti_to_stl

HERE = Path(__file__).resolve().parent

# You define the names. Here: the bundled 5-label lung-lobe sample.
LABEL_NAMES = {
    1: "left_lower_lobe",
    2: "left_upper_lobe",
    3: "right_lower_lobe",
    4: "right_middle_lobe",
    5: "right_upper_lobe",
}


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "data" / "lung_lobe_seg.nii.gz")
    out = sys.argv[2] if len(sys.argv) > 2 else "out"
    mode = sys.argv[3] if len(sys.argv) > 3 else "csg"

    written = nifti_to_stl(inp, out, mode=mode, label_names=LABEL_NAMES, suffix="_3d")
    print(f"Wrote {len(written)} STL file(s) to {out}/:")
    for p in written:
        print("  ", Path(p).name)


if __name__ == "__main__":
    main()
