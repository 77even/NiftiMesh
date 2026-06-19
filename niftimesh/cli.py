# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface: ``niftimesh`` (also ``python -m niftimesh``).

    niftimesh lobe_seg.nii.gz out/ --mode csg --label-names names.json
    niftimesh organs.nii.gz out/ --mode independent --suffix _3d
    niftimesh seg.nii.gz out_raw/ --mode naive        # baseline marching cubes
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ._version import __version__
from .convert import nifti_to_stl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="niftimesh",
        description="Reconstruct per-label STL surfaces from a NIfTI segmentation "
                    "(boolean-CSG peel or independent organs).")
    p.add_argument("input", help="Input .nii / .nii.gz segmentation, or a directory of them.")
    p.add_argument("output", help="Output directory for the per-label STL files.")
    p.add_argument("--mode", choices=["csg", "independent", "naive"], default="csg",
                   help="Reconstruction mode (default: csg).")
    p.add_argument("--label-names", default=None,
                   help="JSON file with a {label_value: name} map. "
                        "Labels with no entry are named label_<value>.")
    p.add_argument("--gaussian-sigma", type=float, default=1.5,
                   help="Smoothing radius in mm (default: 1.5).")
    p.add_argument("--suffix", default="",
                   help="Suffix appended to each STL filename stem (e.g. '_3d').")
    p.add_argument("--nthreads", type=int, default=None, help="Worker threads.")
    p.add_argument("--ascii", action="store_true", help="Write ASCII STL instead of binary.")
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    p.add_argument("--version", action="version", version=f"niftimesh {__version__}")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s")

    label_names = None
    if args.label_names:
        with open(args.label_names) as f:
            label_names = {int(k): v for k, v in json.load(f).items()}

    in_path = Path(args.input)
    inputs = []
    if in_path.is_dir():
        inputs = sorted(in_path.glob("*.nii.gz")) + sorted(in_path.glob("*.nii"))
        if not inputs:
            print(f"No .nii/.nii.gz files in {in_path}", file=sys.stderr)
            return 2
    elif in_path.is_file():
        inputs = [in_path]
    else:
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 2

    out_root = Path(args.output)
    total = 0
    for f in inputs:
        out_dir = out_root if len(inputs) == 1 else out_root / f.name.replace(".nii.gz", "").replace(".nii", "")
        written = nifti_to_stl(
            f, out_dir,
            mode=args.mode, label_names=label_names,
            gaussian_sigma_mm=args.gaussian_sigma, suffix=args.suffix,
            nthreads=args.nthreads, binary=not args.ascii)
        total += len(written)
    print(f"Done: {total} STL file(s) written to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
