#!/usr/bin/env python3
"""Render the README before/after comparison from ONE multi-label NIfTI.

Three voxel-to-STL pipelines, identical input, identical cameras:

  * Naive marching cubes -- generic voxel->STL: raw staircase, faceted.
  * 3D Slicer (emulated) -- per-segment Flying-Edges + light smoothing +
    decimation (the typical "Export to model" default): smoother than naive,
    but every label is built independently -- lumpy silhouette, no shared seam.
  * NiftiMesh -- watertight, smooth, manifold; CSG mode shares each fissure
    seam bit-identically between neighbouring labels.

Rendering is **pure VTK, faithful to how 3D Slicer displays a dragged-in STL**:
vtkSTLReader -> vtkPolyDataNormals (feature-angle split, smooth point normals)
-> vtkActor (matte Gouraud material) -> vtkLightKit -> blue gradient background
-> offscreen vtkWindowToImageFilter. That is what makes the CSG result look the
way it does in Slicer -- the shared fissure seams render as one continuous
surface, with no crease line. Multiple viewpoints. Nothing is hard-coded to
lungs -- pass any segmentation via --preset / --names.

    python assets/render_comparison.py --input lobe_seg.nii.gz \
        --outdir assets --preset lung_lobe

Writes comparison.png (method x view grid) and closeup.png (3-way surface zoom).
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import vtk
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import niftimesh
from niftimesh.convert import _read_segmentation, reconstruct, save_meshes

# A refined jewel-toned palette for the lung lobes -- vivid and distinct, and
# tuned to pop on Slicer's blue gradient background. Other regions fall back to
# PALETTE, cycled by sorted name for stable colours.
LOBE_COLORS = {
    "right_upper_lobe": "#E36A5E",   # warm coral
    "right_middle_lobe": "#2C7FB8",  # ocean blue
    "right_lower_lobe": "#EDEFF2",   # pearl white
    "left_upper_lobe": "#EBB24E",    # honey amber
    "left_lower_lobe": "#A86CC4",    # soft violet
}
PALETTE = [
    "#E36A5E", "#2C7FB8", "#EDEFF2", "#EBB24E", "#A86CC4",
    "#5AB4A8", "#E58FB0", "#7FA9DC", "#F0A04B", "#B5B9C4",
]
# 3D Slicer's blue 3D-view gradient (sampled from a real Slicer viewport).
BG_TOP = "#767ABF"
BG_BOTTOM = "#B6B9E2"
PANEL = (620, 720)
INK = "#f4f6fb"
SUBINK = "#c9cee0"
CANVAS = "#3b3f57"   # dusk blue-grey, so the gradient panels sit in a set

VIEWS = [  # (label, direction the camera looks FROM, relative to volume centre)
    ("Anterior",  (0.0, -1.0, 0.16)),
    ("Lateral",   (1.0, -0.10, 0.14)),
    ("Posterior", (0.04, 1.0, 0.16)),
]

METHODS = [
    ("naive", "Naive marching cubes", "#cfd6ea"),
    ("slicer", "3D Slicer (default export)", "#cfc4ef"),
    ("niftimesh", "NiftiMesh", "#9fe0ff"),
]


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# -- Slicer-style emulation (assets-only; not part of the package) ---------
def _vtk_image(arr):
    a = np.ascontiguousarray(arr).astype(np.float32)
    img = vtk.vtkImageData()
    img.SetDimensions(a.shape[2], a.shape[1], a.shape[0])
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.GetPointData().SetScalars(numpy_to_vtk(
        a.transpose(2, 1, 0).ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT))
    return img


def build_slicer_like(seg, sigma_vox=0.9, decimate=0.20, smooth_iter=18, passband=0.10):
    """Emulate 3D Slicer's default segmentation -> closed-surface export: per
    label, mild Gaussian + marching cubes + WindowedSinc + Pro decimation.
    Returns {label: vtkPolyData} in IJK (spacing 1,1,1)."""
    labels = sorted(int(l) for l in np.unique(seg) if l != 0)
    out = {}
    for lbl in labels:
        g = gaussian_filter((seg == lbl).astype(np.float32), sigma=sigma_vox, mode="nearest")
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(_vtk_image(g))
        mc.SetValue(0, 0.5)
        mc.Update()
        pd = mc.GetOutput()
        sinc = vtk.vtkWindowedSincPolyDataFilter()
        sinc.SetInputData(pd)
        sinc.SetNumberOfIterations(smooth_iter)
        sinc.SetPassBand(passband)
        sinc.NormalizeCoordinatesOn()
        sinc.Update()
        pd = sinc.GetOutput()
        if decimate > 0 and pd.GetNumberOfPoints() > 0:
            dec = vtk.vtkDecimatePro()
            dec.SetInputData(pd)
            dec.SetTargetReduction(decimate)
            dec.PreserveTopologyOn()
            dec.Update()
            pd = dec.GetOutput()
        if pd.GetNumberOfPoints() > 0:
            out[lbl] = pd
    return out


# -- mesh generation -------------------------------------------------------
def _gen(input_path, tmp, mode, names):
    seg, spacing, origin = _read_segmentation(input_path)
    sets = {
        "naive": reconstruct(seg, spacing, mode="naive"),
        "slicer": build_slicer_like(seg),
        "niftimesh": reconstruct(seg, spacing, mode=mode),
    }
    out = {}
    for key, meshes in sets.items():
        d = Path(tmp) / key
        out[key] = save_meshes(meshes, d, spacing, origin, label_names=names)
    return out


def _load(paths):
    """Load each STL into a vtkPolyData (vtkSTLReader, merging on)."""
    out = {}
    for p in paths:
        r = vtk.vtkSTLReader()
        r.SetFileName(str(p))
        r.Update()
        out[Path(p).stem] = r.GetOutput()
    return out


def _color_map(names):
    out, j = {}, 0
    for n in sorted(set(names)):
        if n in LOBE_COLORS:
            out[n] = LOBE_COLORS[n]
        else:
            out[n] = PALETTE[j % len(PALETTE)]
            j += 1
    return out


def _bounds(polys):
    b = np.array([m.GetBounds() for m in polys.values()])  # (n, 6): xmin,xmax,ymin,...
    return (b[:, 0].min(), b[:, 1].max(), b[:, 2].min(),
            b[:, 3].max(), b[:, 4].min(), b[:, 5].max())


# -- pure-VTK render panel (faithful to Slicer's model display) ------------
def _render_panel(polys, colors, bounds, view_dir, smooth, zoom=1.32,
                  show_edges=False, size=PANEL):
    ren = vtk.vtkRenderer()
    ren.GradientBackgroundOn()
    ren.SetBackground(*_hex(BG_BOTTOM))    # bottom (lighter)
    ren.SetBackground2(*_hex(BG_TOP))      # top (darker)

    for name, pd in polys.items():
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(pd)
        if smooth:
            # Slicer's display normals: split at sharp fissure rims, smooth
            # (Gouraud) everywhere else -> the seamless lobe look.
            nrm.SetFeatureAngle(30)
            nrm.SplittingOn()
            nrm.ComputePointNormalsOn()
        else:
            nrm.ComputePointNormalsOff()
            nrm.ComputeCellNormalsOn()
        nrm.ConsistencyOn()
        nrm.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(nrm.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*_hex(colors.get(name, "#cccccc")))
        if smooth:
            prop.SetInterpolationToGouraud()
        else:
            prop.SetInterpolationToFlat()   # naive: show the voxel facets
        # Slicer model material defaults (matte); a hint of specular for life.
        prop.SetAmbient(0.10)
        prop.SetDiffuse(0.95)
        prop.SetSpecular(0.12)
        prop.SetSpecularPower(15)
        if show_edges:
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(*_hex("#5b6170"))
            prop.SetLineWidth(0.4)
        ren.AddActor(actor)

    # vtkLightKit -- 3D Slicer's lighting. Lights follow the camera, so the
    # framing is consistent across views and the shared seams stay shadow-free.
    lk = vtk.vtkLightKit()
    lk.AddLightsToRenderer(ren)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(size[0], size[1])
    rw.SetMultiSamples(8)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    c = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    d = np.array(view_dir, float)
    d /= np.linalg.norm(d)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(*(c + d))            # direction only; ResetCamera sets distance
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera(xmin, xmax, ymin, ymax, zmin, zmax)
    cam.Zoom(zoom)
    ren.ResetCameraClippingRange()

    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    vimg = w2i.GetOutput()
    w, h, _ = vimg.GetDimensions()
    arr = vtk_to_numpy(vimg.GetPointData().GetScalars()).reshape(h, w, -1)
    return arr[::-1]   # VTK origin is bottom-left


# -- compositing (PIL) -----------------------------------------------------
def _font(size):
    for c in ([
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]):
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _ctext(d, cx, y, text, font, fill):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text((cx - (r - l) / 2, y), text, font=font, fill=fill)


def _ltext(d, x, cy, text, font, fill):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text((x, cy - (b - t) / 2), text, font=font, fill=fill)


def render_grid(meshes, outdir, mode):
    cols = [_load(meshes[k]) for k, *_ in METHODS]
    colors = _color_map([n for c in cols for n in c])
    bounds = _bounds(cols[-1])

    pw, ph = PANEL
    gap, margin, gutter, head = 10, 44, 96, 104
    ncol, nrow = 3, len(VIEWS)
    W = margin * 2 + gutter + ncol * pw + (ncol - 1) * gap
    H = head + nrow * ph + (nrow - 1) * gap + margin
    canvas = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(canvas)

    x0 = margin + gutter
    for ci, (_, title, tc) in enumerate(METHODS):
        cx = x0 + ci * (pw + gap) + pw // 2
        _ctext(d, cx, 48, title, _font(27), tc)

    for ri, (vlabel, vdir) in enumerate(VIEWS):
        y = head + ri * (ph + gap)
        _ltext(d, margin, y + ph // 2, vlabel, _font(20), SUBINK)
        for ci, (key, *_r) in enumerate(METHODS):
            x = x0 + ci * (pw + gap)
            img = _render_panel(cols[ci], colors, bounds, vdir, smooth=(key != "naive"))
            canvas.paste(Image.fromarray(img), (x, y))

    out = Path(outdir) / "comparison.png"
    canvas.save(out)
    print("wrote", out)


def render_closeup(meshes, outdir):
    nm = _load(meshes["niftimesh"])
    big = max(nm.items(), key=lambda kv: kv[1].GetNumberOfPoints())[0]
    loaded = {k: _load(meshes[k]) for k, *_ in METHODS}
    colors = _color_map(list(nm))
    color = colors[big]

    pw, ph = 680, 760
    gap, margin, head = 10, 44, 100
    W = margin * 2 + 3 * pw + 2 * gap
    H = head + ph + 58 + margin
    canvas = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(canvas)
    _ctext(d, W // 2, 44, "Surface detail — same region, same camera", _font(30), INK)

    bounds = nm[big].GetBounds()
    for ci, (key, title, tc) in enumerate(METHODS):
        x = margin + ci * (pw + gap)
        img = _render_panel({big: loaded[key][big]}, {big: color}, bounds, VIEWS[1][1],
                            smooth=(key != "naive"), zoom=1.74,
                            show_edges=(key == "naive"), size=(pw, ph))
        canvas.paste(Image.fromarray(img), (x, head))
        _ctext(d, x + pw // 2, head + ph + 14, title, _font(23), tc)

    out = Path(outdir) / "closeup.png"
    canvas.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="multi-label NIfTI segmentation")
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--mode", choices=["csg", "independent"], default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--names", default=None, help="JSON {label_value: name}")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    names, mode = None, args.mode
    if args.preset:
        p = niftimesh.get_preset(args.preset)
        names = dict(p.labels)
        mode = mode or p.mode
    if args.names:
        import json
        names = {int(k): v for k, v in json.load(open(args.names)).items()}
    mode = mode or "csg"

    with tempfile.TemporaryDirectory() as tmp:
        meshes = _gen(args.input, tmp, mode, names)
        render_grid(meshes, args.outdir, mode)
        render_closeup(meshes, args.outdir)


if __name__ == "__main__":
    main()
