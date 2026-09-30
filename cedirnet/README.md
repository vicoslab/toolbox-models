# CeDiRNet Toolbox integration

## Images without objects

An image with no objects is a valid training sample. Submit the annotation
without drawing any points or vectors; export writes `points: []`. No additional
labeling control is required. Images with no submitted annotation remain
unlabeled and are skipped. A submitted annotation is treated as complete: do not
submit a partial annotation if objects are still unmarked.

Alternatively, provide negative images explicitly in a manifest:

```json
{
  "train": [
    {"image_path": "positive.png", "points": [[120, 80]]},
    {"image_path": "negative.png", "points": []}
  ],
  "test": []
}
```

The existing sine/cosine direction outputs regress toward **(0,0)** on negative
images, over non-ignored pixels. No learned positive/negative mask, additional
head, or detection-loss objective is introduced. Positive images retain their
nearest-center targets, weights and gradients.

Distance to a nonexistent center is undefined. `omit_negative_distance_loss()`
in `train.py` excludes only that image's center-distance component from the upstream loss
and its diagnostics. Existing orientation attributes are foreground-only. The
policy uses generated direction targets after augmentation, not the instance
support raster or padded center coordinates, so border positives retain their
original losses. Loss normalization and the localizer's thresholds and
normalization are unchanged. A null direction field does not itself guarantee
zero low-score localizer candidates.

These changes are plugin-local: no additional upstream patch or setup-script
change is required. Reload/update the model's plugin files before starting a
new training job; existing workers keep their already imported code.

## Regression tests

Use a Python environment with the model's dependencies:

```bash
python -m unittest discover -s cedirnet/tests -p 'test_*.py'
node --test cedirnet/tests/test_ui.mjs
```

To also exercise actual upstream dataset, target and loss code, point
`CEDIRNET_SOURCE` at the CeDiRNet-3DoF checkout installed by `setup.sh` (with the
existing plugin patches applied):

```bash
CEDIRNET_SOURCE="$TOOLBOX_CACHE/cedirnet" \
python -m unittest discover -s cedirnet/tests -p 'test_negative_training.py' -v
```

The integration checks cover singleton/all-negative batches, mixed batches in
both orders, unchanged positive gradients, signed null-vector gradients,
ignored pixels, exact-zero directions with arbitrary distance, and geometric
augmentation. They also exercise the existing foreground-only orientation
criterion; this does not enable orientation in the point-only training CLI.
