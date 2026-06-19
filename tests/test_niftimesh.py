"""Core tests. The `independent` and `naive` modes need only numpy/scipy/vtk;
the `csg` test is skipped when meshlib is not installed."""

import numpy as np
import pytest

import niftimesh
from niftimesh import build_naive, get_preset, reconstruct
from niftimesh.presets import PRESETS


def _two_blocks():
    """A 40^3 volume with two adjacent labelled blocks (share a face)."""
    seg = np.zeros((40, 40, 40), dtype=np.uint8)
    seg[8:32, 8:32, 8:20] = 1
    seg[8:32, 8:32, 20:32] = 2
    return seg, (1.0, 1.0, 1.0)


def _poly_ok(poly):
    return poly is not None and poly.GetNumberOfPoints() > 0 and poly.GetNumberOfCells() > 0


# -- package surface -------------------------------------------------------
def test_version():
    assert isinstance(niftimesh.__version__, str)
    assert niftimesh.__version__.count(".") >= 2


def test_presets_well_formed():
    for name, preset in PRESETS.items():
        assert preset.mode in ("csg", "independent")
        assert preset.labels and all(isinstance(k, int) for k in preset.labels)
    assert get_preset("lung_lobe").mode == "csg"
    assert get_preset("core_organs").mode == "independent"


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("does_not_exist")


# -- reconstruction modes --------------------------------------------------
def test_naive_per_label():
    seg, spacing = _two_blocks()
    meshes = build_naive(seg, spacing)
    assert set(meshes) == {1, 2}
    assert all(_poly_ok(p) for p in meshes.values())


def test_independent_mode():
    seg, spacing = _two_blocks()
    meshes = reconstruct(seg, spacing, mode="independent")
    assert set(meshes) == {1, 2}
    assert all(_poly_ok(p) for p in meshes.values())


def test_empty_volume_returns_empty():
    seg = np.zeros((16, 16, 16), dtype=np.uint8)
    assert reconstruct(seg, (1, 1, 1), mode="independent") == {}
    assert build_naive(seg, (1, 1, 1)) == {}


def test_bad_mode_raises():
    seg, spacing = _two_blocks()
    with pytest.raises(ValueError):
        reconstruct(seg, spacing, mode="nope")


@pytest.mark.skipif(
    pytest.importorskip("meshlib", reason="csg mode needs meshlib") is None,
    reason="meshlib missing")
def test_csg_mode_shares_seam():
    seg, spacing = _two_blocks()
    meshes = reconstruct(seg, spacing, mode="csg")
    assert set(meshes) == {1, 2}
    assert all(_poly_ok(p) for p in meshes.values())


# -- file I/O round trip ---------------------------------------------------
def test_save_meshes_writes_named_files(tmp_path):
    seg, spacing = _two_blocks()
    meshes = reconstruct(seg, spacing, mode="independent")
    written = niftimesh.save_meshes(
        meshes, tmp_path, spacing, origin=(0, 0, 0),
        label_names={1: "block_a", 2: "block_b"}, suffix="_3d")
    names = {p.split("/")[-1] for p in written}
    assert names == {"block_a_3d.stl", "block_b_3d.stl"}
    for p in written:
        assert (tmp_path / p.split("/")[-1]).stat().st_size > 0
