# Boolean-CSG ("mimics") lobe / segment STL reconstruction.
#
# Splits a closed organ hull with per-label half-space cutters via meshlib
# boolean CSG so every region is simultaneously
#   * solid (no cavity), watertight (boundary edges = 0),
#   * manifold (no orange streaks in one-sided viewers),
#   * self-intersection-free (no folds), and
#   * sharing its fissure seam bit-identically with each neighbour (no crack,
#     no sub-pixel black line) -- except a few sub-millimetre triple-junction
#     T-points that are invisible outside extreme zoom.
#
# Mechanism (peel one label at a time):
#     remaining = closed hull (union, gaussian sigma 1.5 mm, iso 0.5, ws40)
#     for L in order[:-1]:
#         cutter_L = MC( stack_L - max_{k in rest} stack_k = 0 )   # closed solid
#         seg[L]    = Intersection(remaining, cutter_L)            # peeled lobe
#         remaining = DifferenceAB(remaining, cutter_L)            # the rest
#     seg[order[-1]] = remaining
# Intersection and DifferenceAB of the SAME (remaining, cutter_L) pair come from
# one (deterministic) meshlib corefinement, so the two sides share the cut
# surface vertex-for-vertex -- that is what makes the seam seamless.
#
# Everything is built in physical-mm space; the caller applies the IJK->World
# translation (origin). The field gaussian uses per-axis voxel sigma so the
# physical-mm smoothing stays isotropic regardless of anisotropic spacing.
#
# Needs `meshlib` (CPU-only, `pip install meshlib`). Import is lazy so the rest
# of the package imports without it; build() raises a clear error if missing.

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import vtk
from scipy.ndimage import gaussian_filter, generate_binary_structure, label as scipy_label
from vtk.util.numpy_support import (
    numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy,
)

logger = logging.getLogger(__name__)


class CSGMeshBuilder:
    """Reconstruct one closed STL surface per non-zero label.

    Two reconstruction modes:

    * :meth:`build` (``mode="csg"``) -- boolean-CSG peel. Splits ONE connected
      hull into per-label solids with half-space cutters; adjacent labels share
      their interface seam bit-identically (no crack / black line). For tasks
      that partition a single structure by internal interfaces: lung lobes /
      segments, couinaud liver segments.
    * :meth:`build_independent` (``mode="independent"``) -- each label built as
      its OWN closed smoothed surface (no peel, no shared cut). For DISJOINT
      multi-organ scenes (liver + spleen + kidneys + ...), so a separate organ
      is never sliced flat against a touching neighbour.

    Usage::

        meshes = CSGMeshBuilder(gaussian_sigma_mm=1.5).build(seg, spacing)
        # -> {int(label): vtkPolyData} in full-volume IJK coords, outward-oriented.

    Cutters / gaussian are mutually independent and VTK + meshlib both release
    the GIL, so thread-parallelising them scales (~2.7x cutters, ~4.9x gaussian
    on 10 cores). The peel booleans stay sequential (each depends on ``remaining``).
    """

    NTHREADS = min(8, os.cpu_count() or 1)

    def __init__(self, gaussian_sigma_mm: float = 1.5,
                 nthreads: int = NTHREADS, peel_order=None):
        """
        Args:
            gaussian_sigma_mm: physical smoothing sigma in mm (default 1.5).
            nthreads: thread pool size for the independent gaussian / cutter work.
            peel_order: optional explicit label peel order; default = sorted labels
                (anatomically reasonable for lobes / segments).
        """
        self.gaussian_sigma_mm = gaussian_sigma_mm
        self.nthreads = max(1, nthreads)
        self.peel_order = peel_order
        self._mm = None
        self._mn = None

    # -- meshlib (lazy) ----------------------------------------------------
    def _ensure_meshlib(self):
        """Import meshlib lazily, raising a clear actionable error if missing."""
        if self._mm is None:
            try:
                from meshlib import mrmeshnumpy as mn
                from meshlib import mrmeshpy as mm
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "CSG reconstruction (mode='csg') needs meshlib -- "
                    "`pip install meshlib`") from exc
            self._mm, self._mn = mm, mn
        return self._mm, self._mn

    def _to_mm(self, pd):
        return self._mn.meshFromFacesVerts(
            self._cells_of(pd).astype(np.int32), self._pts_of(pd).astype(np.float64))

    def _from_mm(self, mesh):
        return (np.asarray(self._mn.getNumpyVerts(mesh)),
                np.asarray(self._mn.getNumpyFaces(mesh.topology)))

    # -- static VTK / numpy helpers ----------------------------------------
    @staticmethod
    def _mc_iso(vol, iso, vox_spacing=(1.0, 1.0, 1.0)):
        """Marching cubes of a float field at `iso`. `vol` is (z, y, x).
        vox_spacing=(sx,sy,sz) sets the MC grid spacing: (1,1,1) -> IJK voxel
        coords; the real image spacing -> physical-mm (isotropic) coords. build()
        uses the latter so the meshlib booleans run in mm space -- IJK space
        distorts anisotropic data and makes the CSG booleans mis-classify on
        multi-component hulls (e.g. separated L/R lungs) and drop labels."""
        arr = np.ascontiguousarray(vol).astype(np.float32)
        img = vtk.vtkImageData()
        img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
        img.SetSpacing(vox_spacing[0], vox_spacing[1], vox_spacing[2])
        img.SetOrigin(0.0, 0.0, 0.0)
        img.GetPointData().SetScalars(numpy_to_vtk(
            arr.transpose(2, 1, 0).ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT))
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, iso)
        mc.ComputeNormalsOff()
        mc.Update()
        return mc.GetOutput()

    @staticmethod
    def _ws40(pd):
        """Light WindowedSinc smoothing (40 iter, passband 0.05) -- matches the
        validated lobe pipeline; most of the smoothness comes from the gaussian
        field, this only de-staircases."""
        ws = vtk.vtkWindowedSincPolyDataFilter()
        ws.SetInputData(pd)
        ws.SetNumberOfIterations(40)
        ws.SetPassBand(0.05)
        ws.NormalizeCoordinatesOn()
        ws.BoundarySmoothingOff()
        ws.FeatureEdgeSmoothingOff()
        ws.NonManifoldSmoothingOn()
        ws.Update()
        return ws.GetOutput()

    @staticmethod
    def _cells_of(pd):
        return vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int64)

    @staticmethod
    def _pts_of(pd):
        return vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)

    @staticmethod
    def _make_pd(pts, tris):
        out = vtk.vtkPolyData()
        vp = vtk.vtkPoints()
        vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts, np.float64), deep=True))
        out.SetPoints(vp)
        n = len(tris)
        flat = np.hstack([np.full((n, 1), 3, np.int64), tris]).ravel()
        ca = vtk.vtkCellArray()
        ca.SetCells(n, numpy_to_vtkIdTypeArray(np.ascontiguousarray(flat), deep=True))
        out.SetPolys(ca)
        return out

    @classmethod
    def _consistent_outward(cls, V, F):
        """Consistent winding + outward normals via signed-volume sign (flip whole
        mesh if negative). More robust than vtkPolyDataNormals AutoOrient, which
        flips some concave lobes inward (-> dark/orange in one-sided viewers)."""
        o = cls._make_pd(V, F)
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(o)
        nf.ConsistencyOn()
        nf.AutoOrientNormalsOff()
        nf.SplittingOff()
        nf.ComputePointNormalsOn()
        nf.Update()
        o = nf.GetOutput()
        P = cls._pts_of(o)
        Fc = cls._cells_of(o)
        vol = np.einsum("ij,ij->i", P[Fc[:, 0]], np.cross(P[Fc[:, 1]], P[Fc[:, 2]])).sum() / 6.0
        if vol < 0:
            o = cls._make_pd(P, Fc[:, [0, 2, 1]])
            nf = vtk.vtkPolyDataNormals()
            nf.SetInputData(o)
            nf.ConsistencyOn()
            nf.AutoOrientNormalsOff()
            nf.SplittingOff()
            nf.ComputePointNormalsOn()
            nf.Update()
            o = nf.GetOutput()
        return o

    @staticmethod
    def _largest_component(pd):
        """Keep only the largest connected component. The first peeled label can
        pick up tiny cutter-noise blobs (a few stray triangles + non-manifold
        edges) from Intersection(hull, cutter); a lobe/segment is single-piece,
        so this drops the noise without touching the main body."""
        cc = vtk.vtkPolyDataConnectivityFilter()
        cc.SetInputData(pd)
        cc.SetExtractionModeToLargestRegion()
        cc.Update()
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(cc.GetOutput())
        cl.Update()
        return cl.GetOutput()

    # -- public API --------------------------------------------------------
    def build(self, segmentation, spacing) -> dict:
        """Boolean-CSG reconstruct one closed STL surface per non-zero label.

        Args:
            segmentation: integer label volume, numpy (z, y, x).
            spacing: reference image spacing (sx, sy, sz), e.g. from
                SimpleITK ``GetSpacing()``.

        Returns:
            {int(label): vtkPolyData} in FULL-volume IJK voxel coords, outward-
            oriented. Caller applies the IJK->World transform. Empty dict if
            there are no foreground labels.
        """
        seg = np.ascontiguousarray(segmentation)
        if not seg.any():
            return {}
        # Separated multi-component hulls (e.g. left/right lungs split by an air
        # gap) MUST be peeled independently: meshlib's DifferenceAB drops the
        # A-component that doesn't touch the cutter, so a long peel chain eats the
        # second blob away entirely (18-segment lungs lost the whole right lung).
        # Split by foreground connected component into single-component hulls
        # (which never trigger that), peel each on its own. A single-component
        # volume degenerates to one plain peel.
        labels = sorted(int(lbl) for lbl in np.unique(seg) if lbl != 0)
        cc, ncomp = scipy_label(seg > 0, structure=generate_binary_structure(3, 1))
        if ncomp <= 1:
            out = self._build_one(seg, spacing)
        else:
            out = {}
            for c in range(1, ncomp + 1):
                out.update(self._build_one(np.where(cc == c, seg, 0).astype(seg.dtype), spacing))
        # Fail-safe warning: a meshlib boolean can still fail on pathological
        # geometry and silently drop a label. Warn loudly so a missing segment is
        # never discovered only by counting output files (no fallback -- the
        # caller gets the CSG result as-is).
        missing = [L for L in labels if L not in out]
        if missing:
            logger.warning(
                "CSG peel produced %d/%d labels; dropped %s (likely a meshlib "
                "boolean failure on pathological geometry).", len(out), len(labels), missing)
        return out

    def _build_one(self, segmentation, spacing) -> dict:
        """Peel ONE connected hull (single foreground component) into per-label
        closed solids. Same {label: vtkPolyData} IJK contract as build();
        build() splits multi-component volumes and calls this per component."""
        mm, _ = self._ensure_meshlib()

        seg = np.ascontiguousarray(segmentation)
        labels = sorted(int(lbl) for lbl in np.unique(seg) if lbl != 0)
        if not labels:
            return {}

        # Crop to the foreground bbox (+8 voxel pad) -- the organ is a fraction of
        # the CT, so this is the dominant speedup. Vertices are shifted back by `lo`.
        nz = np.argwhere(seg > 0)
        lo = np.maximum(nz.min(0) - 8, 0)
        hi = np.minimum(nz.max(0) + 9, np.array(seg.shape))
        seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        # Per-axis voxel sigma so physical-mm smoothing is isotropic. spacing is
        # (x, y, z); numpy axes are (z, y, x).
        s = self.gaussian_sigma_mm
        sv = [s / spacing[2], s / spacing[1], s / spacing[0]]

        li = {lbl: i for i, lbl in enumerate(labels)}
        order = list(self.peel_order) if self.peel_order is not None else list(labels)

        with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
            # Parallel gaussian (union + per-label, all independent).
            masks = [(seg > 0).astype(np.float32)] + \
                    [(seg == lbl).astype(np.float32) for lbl in labels]
            gs = list(ex.map(lambda m: gaussian_filter(m, sigma=sv, mode="nearest"), masks))
            us = gs[0]
            stack = np.stack(gs[1:])
            # Suffix-max so each cutter field is one subtraction (avoids re-maxing
            # over `rest` every peel): suf[i] = max_{k>=i} stack[order[k]].
            suf = [None] * (len(order) + 1)
            for i in range(len(order) - 1, -1, -1):
                suf[i] = stack[li[order[i]]] if suf[i + 1] is None \
                    else np.maximum(stack[li[order[i]]], suf[i + 1])
            fields = []
            for i in range(len(order) - 1):  # cutter(order[i]) = {stack_L > max(rest)}
                f = (stack[li[order[i]]] - suf[i + 1]).copy()
                f[0, :, :] = f[-1, :, :] = -1   # pad boundary negative -> closed solid {f>0}
                f[:, 0, :] = f[:, -1, :] = -1
                f[:, :, 0] = f[:, :, -1] = -1
                fields.append(f)
            # Parallel MC + ws40 + to_mm: hull at iso 0.5, cutters at iso 0.
            # MC in physical-mm space (real spacing) so the meshlib booleans run
            # isotropic -- IJK space mis-classifies on multi-component hulls.
            allf = [(us, 0.5)] + [(f, 0.0) for f in fields]
            meshes = list(ex.map(
                lambda p: self._to_mm(self._ws40(self._mc_iso(p[0], p[1], spacing))), allf))
            # ws40 occasionally folds a complex cutter into a few self-intersecting
            # triangles. Left in, DifferenceAB transfers them onto `remaining`, and
            # the NEXT boolean then fails ("mesh A has self-intersections") -- which
            # silently drops every remaining label (e.g. 18-segment lung lost RS5+).
            # Fix any self-intersecting cutter/hull at the source; the same fixed
            # mesh feeds both Intersection and DifferenceAB, so the shared cut is
            # preserved. No-op for the common clean case.
            #   localFixSelfIntersections(CutAndFill): re-meshes ONLY the
            #   self-intersecting patch -- ~5x faster than the whole-mesh voxel
            #   remesh and clears it (the default Relax method leaves residue).
            #   Whole-mesh voxel fix is the fallback if anything survives.
            # Detect in parallel (releases the GIL); fix serially (each fix is
            # internally multi-threaded, so a pool around it only contends).
            vox = 0.5 * min(spacing)
            si_settings = mm.SelfIntersections.Settings()
            si_settings.method = mm.SelfIntersections.Settings.Method.CutAndFill
            si_counts = list(ex.map(lambda m: mm.findSelfCollidingTriangles(m).size(), meshes))
        for m, n in zip(meshes, si_counts):
            if n > 0:
                mm.localFixSelfIntersections(m, si_settings)
                if mm.findSelfCollidingTriangles(m).size() > 0:
                    mm.fixSelfIntersections(m, vox)
        hull = meshes[0]
        cutmesh = meshes[1:]

        # Sequential peel.
        raw = {}
        remaining = hull
        for i, L in enumerate(order[:-1]):
            c = cutmesh[i]  # same cutter both sides -> seam shared vertex-for-vertex
            raw[L] = self._from_mm(mm.boolean(remaining, c, mm.BooleanOperation.Intersection).mesh)
            remaining = mm.boolean(remaining, c, mm.BooleanOperation.DifferenceAB).mesh
        raw[order[-1]] = self._from_mm(remaining)

        # Verts came out in physical-mm (MC used real spacing); divide by spacing
        # back to cropped-IJK, shift to full-volume IJK ((z,y,x) crop -> (x,y,z)
        # pts), and orient outward -- same IJK contract as build_independent.
        out = {}
        inv = np.array([1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]])
        off = np.array([lo[2], lo[1], lo[0]], dtype=np.float64)
        for L in labels:
            V, F = raw[L]
            if len(V) == 0 or len(F) == 0:
                continue
            # Largest component drops tiny cutter-noise blobs (mainly on the
            # first peeled label); a lobe/segment is single-piece.
            out[int(L)] = self._largest_component(self._consistent_outward(V * inv + off, F))
        return out

    def build_independent(self, segmentation, spacing) -> dict:
        """Build each label as its OWN closed smoothed surface -- no peel, no
        shared cut, no meshlib boolean.

        For DISJOINT multi-organ (liver + spleen + kidneys + ...): every organ
        comes out a complete rounded watertight surface, never sliced flat
        against a touching neighbour -- unlike build()'s CSG peel, which cuts a
        shared plane between adjacent labels and makes separate organs look
        "chopped". Labels that abut keep their full smoothed extent and may
        overlap by ~sigma at the contact (intentional -- independent organs
        shouldn't share a cut plane). Each label's field is closed at the volume
        faces so organs truncated by the scan FOV stay watertight.

        Returns {int(label): vtkPolyData} in full-volume IJK coords, outward-
        oriented (same contract as build()).

        Note: this mode needs only numpy/scipy/vtk -- no meshlib.
        """
        seg = np.ascontiguousarray(segmentation)
        labels = sorted(int(lbl) for lbl in np.unique(seg) if lbl != 0)
        if not labels:
            return {}

        nz = np.argwhere(seg > 0)
        lo = np.maximum(nz.min(0) - 8, 0)
        hi = np.minimum(nz.max(0) + 9, np.array(seg.shape))
        seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        s = self.gaussian_sigma_mm
        sv = [s / spacing[2], s / spacing[1], s / spacing[0]]

        def _one(lbl):
            f = gaussian_filter((seg == lbl).astype(np.float32), sigma=sv, mode="nearest")
            f[0, :, :] = f[-1, :, :] = 0   # close at volume faces -> watertight even
            f[:, 0, :] = f[:, -1, :] = 0   # when the organ is truncated by the FOV
            f[:, :, 0] = f[:, :, -1] = 0
            pd = self._ws40(self._mc_iso(f, 0.5))
            return self._pts_of(pd), self._cells_of(pd)

        with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
            results = list(ex.map(_one, labels))

        out = {}
        off = np.array([lo[2], lo[1], lo[0]], dtype=np.float64)
        for lbl, (V, F) in zip(labels, results):
            if len(V) == 0 or len(F) == 0:
                continue
            out[int(lbl)] = self._consistent_outward(V + off, F)
        return out
