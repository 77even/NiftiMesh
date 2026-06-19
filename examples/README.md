# Examples

`data/lung_lobe_seg.nii.gz` is a 5-label pulmonary-lobe segmentation (238×512×512,
~211 KB) used by the examples and the README figures.

### Convert to STL

```bash
pip install "niftimesh[csg]"
python examples/convert.py                       # bundled sample, CSG lobe preset
python examples/convert.py data/lung_lobe_seg.nii.gz out_indep independent
```

or straight from the CLI:

```bash
niftimesh examples/data/lung_lobe_seg.nii.gz out/ --preset lung_lobe --suffix _3d
```

### Reproduce the comparison figures

```bash
pip install "niftimesh[all]"
python assets/render_comparison.py \
    --input examples/data/lung_lobe_seg.nii.gz --outdir assets --preset lung_lobe
```
