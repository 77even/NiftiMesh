#!/usr/bin/env python3
"""Minimal end-to-end example: a lung-lobe segmentation -> per-label STL files.

    python examples/convert.py            # uses the bundled sample
    python examples/convert.py my_seg.nii.gz out_dir --mode independent
"""

import sys
from pathlib import Path

from niftimesh import nifti_to_stl

HERE = Path(__file__).resolve().parent


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "data" / "lung_lobe_seg.nii.gz")
    out = sys.argv[2] if len(sys.argv) > 2 else "out"
    mode = sys.argv[3] if len(sys.argv) > 3 else None

    if mode:
        written = nifti_to_stl(inp, out, mode=mode, suffix="_3d")
    else:
        # The bundled sample is lung lobes -> use the preset (CSG + lobe names).
        written = nifti_to_stl(inp, out, preset="lung_lobe", suffix="_3d")

    print(f"Wrote {len(written)} STL file(s) to {out}/:")
    for p in written:
        print("  ", Path(p).name)


if __name__ == "__main__":
    main()
