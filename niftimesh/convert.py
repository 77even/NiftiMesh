# Copyright (c) 2026 Justin
# SPDX-License-Identifier: MIT

# High-level NIfTI-segmentation -> per-label STL conversion.
#
# Wraps CSGMeshBuilder (csg / independent) and the naive baseline behind one
# function. Meshes are built in IJK voxel space; this module applies the
# IJK->World transform  world = origin + spacing .* ijk  -- ignoring the DICOM
# direction cosines on purpose (the "mimics" convention that keeps every label
# aligned with the vessel / lobe STLs an editor expects).

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import vtk

from .builder import CSGMeshBuilder
from .naive import build_naive

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]


# -- low-level mesh I/O ----------------------------------------------------
def _ijk_to_world_matrix(spacing, origin) -> "vtk.vtkMatrix4x4":
    m = vtk.vtkMatrix4x4()
    m.Identity()
    for j in range(3):
        m.SetElement(j, j, spacing[j])
        m.SetElement(j, 3, origin[j])
    return m


def _write_polydata(poly, path: str, ijk_to_world, binary: bool = True) -> None:
    if poly is None or poly.GetNumberOfPoints() == 0:
        return
    transform = vtk.vtkTransform()
    transform.SetMatrix(ijk_to_world)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(transform)
    tf.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(tf.GetOutput())
    writer.SetFileName(path)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.Write()


def save_meshes(
    meshes: dict,
    output_dir: PathLike,
    spacing,
    origin=(0.0, 0.0, 0.0),
    label_names: Optional[dict] = None,
    suffix: str = "",
    binary: bool = True,
) -> list:
    """Write a {label: vtkPolyData} dict (IJK coords) to per-label STL files.

    Args:
        meshes: {int label: vtkPolyData} in full-volume IJK voxel coords.
        output_dir: directory for the .stl files (created if needed).
        spacing: (sx, sy, sz) image spacing.
        origin: (ox, oy, oz) image origin (world translation).
        label_names: {int or str label: name}. Missing labels fall back to
            ``label_<value>``.
        suffix: appended to each filename stem (e.g. "_3d" -> "liver_3d.stl").
        binary: write binary STL (default) or ASCII.

    Returns:
        List of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ijk_to_world = _ijk_to_world_matrix(spacing, origin)

    names = {}
    if label_names:
        names = {int(k): v for k, v in label_names.items()}

    written = []
    for lbl, poly in meshes.items():
        if poly is None or poly.GetNumberOfPoints() == 0:
            continue
        name = names.get(int(lbl), f"label_{int(lbl)}")
        path = str(output_dir / f"{name}{suffix}.stl")
        _write_polydata(poly, path, ijk_to_world, binary=binary)
        written.append(path)
    return written


# -- NIfTI loading ---------------------------------------------------------
def _read_segmentation(input_path: PathLike):
    """Load a NIfTI label volume -> (seg (z,y,x) int array, spacing, origin).

    Uses SimpleITK if available (handles .nii/.nii.gz + correct geometry);
    falls back to nibabel.
    """
    input_path = str(input_path)
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(input_path)
        seg = sitk.GetArrayFromImage(img)            # (z, y, x)
        return seg, img.GetSpacing(), img.GetOrigin()
    except ImportError:
        pass

    import nibabel as nib
    img = nib.load(input_path)
    data = np.asarray(img.dataobj)                   # (x, y, z)
    seg = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))  # -> (z, y, x)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    aff = img.affine
    origin = (float(aff[0, 3]), float(aff[1, 3]), float(aff[2, 3]))
    return seg, spacing, origin


# -- public high-level API -------------------------------------------------
def reconstruct(
    segmentation: np.ndarray,
    spacing,
    mode: str = "csg",
    gaussian_sigma_mm: float = 1.5,
    nthreads: Optional[int] = None,
    peel_order=None,
) -> dict:
    """Reconstruct meshes from a label volume in memory.

    Args:
        segmentation: integer label volume, numpy (z, y, x).
        spacing: (sx, sy, sz) image spacing.
        mode: "csg" (boolean-CSG peel, shared seams), "independent" (each label
            its own closed surface), or "naive" (plain marching cubes baseline).
        gaussian_sigma_mm: smoothing radius in mm (csg / independent).
        nthreads: worker threads (default: builder default).
        peel_order: optional explicit CSG peel order.

    Returns:
        {int label: vtkPolyData} in full-volume IJK voxel coords.
    """
    if mode == "naive":
        return build_naive(segmentation, spacing)

    kwargs = {"gaussian_sigma_mm": gaussian_sigma_mm, "peel_order": peel_order}
    if nthreads is not None:
        kwargs["nthreads"] = nthreads
    builder = CSGMeshBuilder(**kwargs)
    if mode == "independent":
        return builder.build_independent(segmentation, spacing)
    if mode == "csg":
        return builder.build(segmentation, spacing)
    raise ValueError(f"Unknown mode '{mode}' (use 'csg', 'independent' or 'naive').")


def nifti_to_stl(
    input_path: PathLike,
    output_dir: PathLike,
    mode: str = "csg",
    label_names: Optional[dict] = None,
    gaussian_sigma_mm: float = 1.5,
    suffix: str = "",
    nthreads: Optional[int] = None,
    binary: bool = True,
) -> list:
    """Convert a NIfTI multi-label segmentation to per-label STL files.

    Args:
        input_path: .nii / .nii.gz segmentation file.
        output_dir: directory for the per-label .stl files.
        mode: "csg" (boolean-CSG peel with shared seams, for one structure split
            by internal interfaces), "independent" (each label its own closed
            surface, for disjoint organs), or "naive" (plain marching cubes).
        label_names: {label_value: name} map for the output filenames. Labels
            with no entry fall back to ``label_<value>``.
        gaussian_sigma_mm: smoothing radius in mm.
        suffix: appended to each STL filename stem.
        nthreads: worker threads.
        binary: write binary STL (default) or ASCII.

    Returns:
        List of written STL file paths.
    """
    seg, spacing, origin = _read_segmentation(input_path)
    meshes = reconstruct(
        seg, spacing, mode=mode,
        gaussian_sigma_mm=gaussian_sigma_mm, nthreads=nthreads)
    written = save_meshes(
        meshes, output_dir, spacing, origin,
        label_names=label_names, suffix=suffix, binary=binary)
    logger.info("NiftiMesh %s: %s -> %d STL files in %s",
                mode, Path(str(input_path)).name, len(written), output_dir)
    return written
