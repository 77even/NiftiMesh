"""NiftiMesh -- watertight, smooth STL reconstruction from NIfTI segmentations.

Two reconstruction modes:

* ``csg``         -- boolean-CSG peel. Splits ONE connected structure by its
                     internal interfaces into per-label solids that share each
                     fissure seam bit-identically (lung lobes / segments,
                     Couinaud liver segments). No cracks, no black seam lines.
* ``independent`` -- each label built as its own closed, smoothed, watertight
                     surface (disjoint multi-organ scenes).

Quick start::

    from niftimesh import nifti_to_stl
    nifti_to_stl("lobe_seg.nii.gz", "out/", preset="lung_lobe")

    # or pick a mode explicitly
    nifti_to_stl("organs.nii.gz", "out/", mode="independent")
"""

from ._version import __version__
from .builder import CSGMeshBuilder
from .convert import nifti_to_stl, reconstruct, save_meshes
from .naive import build_naive
from .presets import PRESETS, Preset, get_preset

__all__ = [
    "__version__",
    "CSGMeshBuilder",
    "nifti_to_stl",
    "reconstruct",
    "save_meshes",
    "build_naive",
    "PRESETS",
    "Preset",
    "get_preset",
]
