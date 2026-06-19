# Copyright (c) 2026 Justin
# SPDX-License-Identifier: MIT

# Naive marching-cubes baseline: the "ordinary" voxel-to-STL conversion every
# generic tool ships. One isosurface per label straight off the binary mask --
# no field smoothing, no surface fairing, no shared seams. It is intentionally
# unpolished: this is the *before* in NiftiMesh's before/after comparison, the
# staircased blocky look you get without the CSG / independent pipeline.

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def _mask_to_polydata(mask, smooth_iter=0):
    """Marching cubes on a binary mask (z, y, x) at iso 0.5, in IJK voxel space.

    Built on the voxel grid (spacing 1,1,1); the caller applies the IJK->World
    transform, matching CSGMeshBuilder so both pipelines share one coordinate
    path. smooth_iter>0 applies a few plain Laplacian iterations (still markedly
    rougher than the WindowedSinc+gaussian field pipeline) -- left at 0 by
    default so the baseline shows the raw voxel staircase.
    """
    arr = np.ascontiguousarray(mask).astype(np.float32)
    img = vtk.vtkImageData()
    img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.GetPointData().SetScalars(numpy_to_vtk(
        arr.transpose(2, 1, 0).ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT))

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, 0.5)
    mc.Update()
    pd = mc.GetOutput()

    if smooth_iter > 0:
        sm = vtk.vtkSmoothPolyDataFilter()
        sm.SetInputData(pd)
        sm.SetNumberOfIterations(smooth_iter)
        sm.Update()
        pd = sm.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def build_naive(segmentation, spacing=None, smooth_iter: int = 0) -> dict:
    """Plain per-label marching cubes -- the generic voxel-to-mesh baseline.

    Args:
        segmentation: integer label volume, numpy (z, y, x).
        spacing: unused (kept for a signature symmetric with CSGMeshBuilder.build);
            meshes are built in IJK and the caller applies spacing via IJK->World.
        smooth_iter: optional plain Laplacian smoothing iterations (default 0).

    Returns:
        {int(label): vtkPolyData} in full-volume IJK voxel coords (same contract
        as CSGMeshBuilder.build). The caller applies the IJK->World transform.
    """
    seg = np.ascontiguousarray(segmentation)
    labels = sorted(int(lbl) for lbl in np.unique(seg) if lbl != 0)
    out = {}
    for lbl in labels:
        pd = _mask_to_polydata(seg == lbl, smooth_iter=smooth_iter)
        if pd.GetNumberOfPoints() > 0:
            out[lbl] = pd
    return out
