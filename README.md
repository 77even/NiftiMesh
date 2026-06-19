<div align="center">

<img src="https://raw.githubusercontent.com/77even/NiftiMesh/main/assets/logo.png" alt="NiftiMesh" width="440">

[![PyPI](https://img.shields.io/pypi/v/niftimesh.svg?cacheSeconds=300)](https://pypi.org/project/niftimesh/)
[![Python](https://img.shields.io/pypi/pyversions/niftimesh.svg?cacheSeconds=300)](https://pypi.org/project/niftimesh/)
[![Downloads](https://img.shields.io/pypi/dm/niftimesh.svg?cacheSeconds=300)](https://pypi.org/project/niftimesh/)
[![Release](https://img.shields.io/github/v/release/77even/NiftiMesh.svg?cacheSeconds=300)](https://github.com/77even/NiftiMesh/releases/latest)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

Turn a multi-label `.nii.gz` segmentation into one clean, closed STL surface per
label — **solid, watertight, manifold, self-intersection-free**. In `csg` mode,
adjacent regions share their interface seam *bit-identically*, so the parts
assemble seamlessly in 3D Slicer / Mimics.

<img src="https://raw.githubusercontent.com/77even/NiftiMesh/main/assets/comparison.png" width="100%">
<img src="https://raw.githubusercontent.com/77even/NiftiMesh/main/assets/closeup.png" width="100%">

*Left → right: naive marching cubes (voxel staircase) · a typical 3D Slicer
default export · NiftiMesh (smooth, watertight, seamless).*

## What it's for

A multi-label segmentation volume → per-label STL meshes. Two modes:

- **`csg`** — for **one structure split by internal interfaces**: lung lobes /
  segments, Couinaud liver segments. Neighbouring labels share their cut seam
  exactly (no crack, no black line).
- **`independent`** — for **disjoint organs** (liver + spleen + kidneys, vessels):
  each label becomes its own closed, smoothed surface.

You define the output names — nothing is hard-coded to any anatomy.

## Install

```bash
pip install "niftimesh[csg]"   # csg mode (needs meshlib)
pip install niftimesh          # independent + naive modes only
```

Python ≥ 3.9. NIfTI is read via `nibabel` (bundled) or `SimpleITK` if installed.
Prebuilt wheels are also attached to each
[GitHub Release](https://github.com/77even/NiftiMesh/releases/latest).

## Usage

```bash
# names from a JSON file: {"1": "left_lower_lobe", "2": "left_upper_lobe", ...}
niftimesh seg.nii.gz out/ --mode csg --label-names names.json --suffix _3d

# omit --label-names to get label_1.stl, label_2.stl, ...
niftimesh organs.nii.gz out/ --mode independent
```

```python
from niftimesh import nifti_to_stl

nifti_to_stl("seg.nii.gz", "out/", mode="csg",
             label_names={1: "left_lower_lobe", 2: "left_upper_lobe"})

# in-memory: numpy (z, y, x) label volume -> {label: vtkPolyData}
from niftimesh import reconstruct
meshes = reconstruct(seg, spacing=(0.84, 0.84, 1.0), mode="independent")
```

## License

[Apache 2.0](LICENSE) © 2026 Justin
