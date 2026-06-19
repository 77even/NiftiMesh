# Examples

`data/lung_lobe_seg.nii.gz` is a 5-label pulmonary-lobe segmentation
(238×512×512, ~211 KB) used as a sample.

```bash
pip install "niftimesh[csg]"

# Python: you define the label names (see convert.py)
python examples/convert.py

# CLI: names from a JSON file ({"1": "left_lower_lobe", ...}), or omit for label_<n>
niftimesh data/lung_lobe_seg.nii.gz out/ --mode csg --label-names names.json --suffix _3d
```
