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

## Inference and validation normalization

Inference, validation, and training-preview detections use the localization
checkpoint's stored BatchNorm statistics, including after a training epoch.
Scores are absolute learned responses, **not probabilities** and not fractions
of the largest heatmap response. The `0.5` display cutoff remains unchanged.
The upstream batch-wide 2000-candidate cap is unchanged; very dense batches can
still affect which candidates are retained even with stable heatmap scores.

The localizer is constructed with statistic buffers even when training requests
`freeze_learning=True`. That training policy still uses batch statistics without
updating the saved evaluation buffers, preserving the original training behavior.
The enclosing estimator remains in training mode for losses: its evaluation path
decodes regression outputs and targets in place. Newly saved checkpoints retain
the evaluation statistics.

Localization checkpoints without valid running mean/variance or learned weights
now fail instead of silently using a partly initialized localizer. Old Toolbox
training checkpoints without those buffers can still initialize the backbone
when the original, separately configured localization checkpoint is provided.
The default training setup supplies that localization checkpoint.

Update/reload the plugin before starting inference or training; already running
workers keep their imported code. No upstream source patch is required.

## Split evaluation and MLflow metrics

At each **Evaluation interval**, and after the final epoch, the trainer evaluates
all labeled images in `train`, `val`, and `test` independently. If `train` is
absent, `data` is used for training and training-set evaluation. This does not
create a held-out split or copy training results into validation/test metrics.
Missing or empty evaluation splits print an explicit skip message.

**Evaluate training set** is enabled by default. Disable it in the training form,
or pass `--evaluate_training false`, to skip only training-set metric calculation.
Training figures remain controlled by **Training samples to visualise**, training
losses are still logged every epoch, and validation/test evaluation is unchanged.
When enabled, metrics cover the entire labeled training set, not just the images
selected for visualization.

MLflow metrics use `training/*`, `validation/*`, and `testing/*`, respectively,
with one-based epoch steps. Training loss components are logged under
`training/*` every epoch; detection precision, recall, F1 at 20 px, localization
errors, point counts and image counts are logged at evaluation epochs.
Orientation-error metrics are omitted: the point-only training CLI does not
provide orientation supervision.
Training-set metrics measure fit, not generalization to unseen images.

Evaluation uses deterministic resize/tensor/normalization transforms, eval mode,
and no gradients. Every eligible image is counted, including negatives and the
last partial batch. **Training samples to visualise** limits only saved training
figures, not metric coverage. Validation/test figures use separate subfolders.
**Evaluation score threshold** applies to all three splits; the CLI identifiers
`display_interval` and `validation_score_threshold` remain unchanged.

Reload/update the plugin before starting a new training job. Existing workers
retain their imported code, and historical runs keep their old metric names.

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

For the real-localizer normalization tests, also supply the setup-installed
localization checkpoint and the Toolbox source (or an installed `modelargs`):

```bash
CEDIRNET_SOURCE="$TOOLBOX_CACHE/cedirnet" \
CEDIRNET_TEST_LOCALIZATION="$TOOLBOX_CACHE/cedirnet/localization_checkpoint.pth" \
TOOLBOX_SOURCE=/absolute/path/to/toolbox \
python -m unittest discover -s cedirnet/tests -p 'test_normalization.py' -v
```

These checks compare an unchanged field alone and in both mixed-batch orders,
verify unchanged training heatmaps and preserved evaluation buffers, exercise
actual validation after training mode, and reject incomplete/nonfinite checkpoints.
Run STEM tests in a separate Python process: the two upstream repositories both
export a package named `models`.
