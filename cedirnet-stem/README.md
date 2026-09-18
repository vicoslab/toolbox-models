# CeDiRNet-STEM: optional particles and semantic segmentation

Enable **nanoparticles**, **multiclass segmentation**, or both. At least one task
must be enabled. Task switches are `"true"`/`"false"` string enums because the
current Toolbox `modelargs` boolean parser rejects `false`; both CLI values are
tested against the host parser.

## Annotation → export → training

1. Create an annotation project using this model's `config.yml`. Group **two**
   registered, equally sized images in **BF, HAADF** order. Use interlaced groups
   with filenames that sort into pairs (e.g. `sample_BF.png`, `sample_HAADF.png`).
   Do not use the current host's broken `divide` grouping mode.
2. Use the two open sidebar panels, **Particle instances** and **Segmentation**.
   Select **PtCo (P)**, then draw particle ellipses (center click first). The exported
   particle label is its alias **nanoparticle**. Paint **Carbon (1)**, **Film (2)**,
   **Vacuum (3)**, or **Ignore (4)**; the segmentation panel also includes Magicwand.
   Regions are shared across the aligned pair. There are no N/S visibility toggles.
3. Submit one annotation per pair. This selected interface intentionally has no
   **No nanoparticles** checkbox. An empty submission could mean a reviewed negative
   or unfinished work: export cannot distinguish them and keeps particle supervision
   missing, just as for segmentation-only annotations. Use an explicitly reviewed
   legacy annotation or an existing reviewed manifest for particle negatives; this
   UI cannot record that confirmation. Resolve multiple annotators before export;
   the adapter rejects multiple annotations rather than guessing a consensus.
4. Assign Train, Validation, or Test splits and use Toolbox **Export**. The host
   imports this plugin's `ls_adapter.export(annotations, export_dir, relpaths,
   shared)` and writes the version-4 manifest. The adapter writes lossless class-ID
   PNGs and attaches their paths and class order to each item. **No manually
   authored manifest or host modification is needed.**
5. Select that manifest for training, enable the tasks you annotated, and set
   **Semantic classes (JSON)** to the same ordered names as the template.

To change semantic classes, edit the ordered `semantic` BrushLabels in the plugin
`config.yml` **before creating the project**, and use the same names/order in the
model option. `Ignore` is reserved and excluded from the class list. Do not change
only the project XML after creation: the host export API does not pass project
configuration to the adapter. Unknown classes are rejected, not reindexed.

### Export semantics

- IDs 0, 1, … follow template class order, not annotation encounter order. Class 0
  is a real class, **not** implicit background. 255 is ignored.
- Unpainted pixels, explicit Ignore, and overlapping *different* classes become
  255. Same-class overlaps are unions. Region order does not change the mask.
- No annotation, an empty submitted annotation, and task-selection-only annotations
  remain **missing supervision**. Only explicit **No nanoparticles** (or legacy
  `reviewed: Nanoparticles`) creates `points: []`. Brush-only annotations do not
  invent particle labels. **Migration:** older empty submissions were treated as
  negatives; review and explicitly confirm them before re-export. Existing exported
  manifests with `points: []` remain valid and unchanged.
- Missing labels for an enabled task fail training. An all-ignore brush is allowed
  and has zero semantic loss; it does not teach the model a background class.
- Shared pair region duplicates are deduplicated. Coordinates remain subpixel
  original-image coordinates; malformed ellipses/vectors, rotated images, unknown
  classes, invalid RLE sizes and inconsistent region dimensions fail export.
- Masks are content-addressed, so later exports/COMBINE cannot overwrite different
  masks under an existing manifest item. Paths are relative to the export folder.

### Supported semantic tools and export diagnosis

| Region/tool | Export coverage |
|---|---|
| BrushLabels | Every configured class and Ignore, RGBA RLE |
| Magicwand | Same-ID `magicwand` + labeled `brushlabels` results; every class and Ignore |
| PolygonLabels / Polygon + Labels | Closed, filled percentage-coordinate polygons |
| RectangleLabels / Rectangle + Labels | Filled rectangles, including clockwise rotation about their top-left corner |
| Semantic EllipseLabels / Ellipse + Labels | Filled ellipses, including rotation about the center |
| Particle EllipseLabels / Ellipse + Labels | Particle `labels` control only; center and mean pixel radius, never a semantic class |

Separate geometry and Labels results are joined using region ID, image target,
and gallery `item_index`. Unlabeled geometry (including a wand with no assigned
class), unknown classes, unsupported spatial types and conflicting paired results
raise errors rather than disappearing. Select a class for every wand region;
press Escape before creating a separate region/class instead of relabeling the
currently selected region. Disjoint regions of the same class are all retained.
The shipped UI remains BrushLabels + Magicwand and the particle ellipse control;
additional semantic geometries support existing/imported LS projects, not new UI
buttons. Class vocabulary still comes from the plugin's semantic BrushLabels.
Polygon fill uses even/odd pixel-center inclusion; rectangle and ellipse masks
also sample pixel centers. Image rotation, open polygons, video sequences, holes
as separate subtractive polygons, keypoints and arbitrary vector segmentation
are not supported. Use brush RLE for raster masks with holes.

Training filters each split for the enabled tasks before reading images: particle-only
requires a `points` key (including an explicit `[]` negative), semantic-only requires
`semantic_mask`, and joint training requires both. A warning names the manifest and
split, reports kept/skipped counts and missing-label counts (which can overlap).
Missing supervision is never converted to a negative. If no usable training samples
remain, training fails clearly; optional validation may be empty. Present malformed
labels, wrong class order, and invalid labeled image pairs still fail validation;
image/mask contents and dimensions are checked when loading a retained sample.

If the warning reports unexpected missing `points`, check that the **exporting** Toolbox model cache
contains this plugin revision, not only the training cache: the older main
exporter reads vector `vertices` only and drops ellipse/semantic results. Then
re-export the reviewed task; existing manifests are not retroactively repaired.
The current combined `ellipselabels` + `nanoparticle` alias schema is supported.
Do not replace missing points with `[]` unless the image is an explicitly confirmed
negative. Segmentation-only/empty submissions must not become particle negatives.

### Ellipse geometry and model compatibility

Label Studio `x/y` are ellipse **center** percentages (not a bounding-box corner);
`radiusX/radiusY` are semi-axis percentages of image width/height. Export converts
them to pixels and writes `[cx, cy, (rx + ry) / 2]`. This arithmetic-mean radius is
an explicit **circle approximation**: ellipse rotation/eccentricity are not trained
or predicted. Rotated ellipses are accepted; image rotation must be reset to zero.
Legacy two-vertex particle exports remain readable. Preannotations now require an
`EllipseLabels` control and encode the existing circle prediction as equal **pixel**
radii (different percentages on non-square images), without clipping at edges.
Update old project XML deliberately before using the new preannotation backend.

The image is nested in a flex layout beside a fixed 380px, two-column sidebar,
with bordered open Collapse panels. The host's create script validates only direct
Image children, so its valueList check does not inspect this nested Image. Group
size **2**, **interlace** creation is tested with the unchanged host and imports
paired `images` tasks correctly; group size 1 would instead create `image` tasks
incompatible with this template. Keep group size 2. No host changes are needed.

SDK parsing uses `alias` before `value`: preannotations use the project's single
configured particle label/alias. Export accepts the plugin's configured serialized
label and legacy `Particle`, rejecting unknown labels (including display-only
`PtCo` when its alias is configured). Change the plugin config before project
creation, not only the project XML, because export receives no project config.

The loader accepts host `train`, `val`, `test`, and `data`. `data` is used for
training only when `train` is absent. Validation uses **only `val`**, never the
held-out `test` split or a training fallback. Review all included training items;
Toolbox does not automatically filter unannotated tasks on export. Legacy version-2
manifests with root-level `semantic_classes` remain readable.

## Model and checkpoints

Input is exactly the upstream `[BF grayscale, HAADF grayscale, zero]` tensor at
0..255 before the FPN's own preprocessing. Paired images must have equal sizes.
Image interpolation is bilinear; class masks use nearest-neighbor. Training flips
are synchronized across modalities, centers, radius targets and semantic masks.

The particle model keeps the stock FPN/ShapeLoss and localization checkpoint layout.
The semantic model instantiates the **same published upstream FPN primitive** with
class logits; its small constructor and ignore-safe cross-entropy + Dice criterion
are plugin-local. **No new upstream branch or research code is required.**

Joint mode intentionally uses **two independent FPNs**. A shared trunk would use
less memory but cannot preserve two independently trained backbone states or allow
independent existing particle/semantic weights to load unchanged. Only enabled
models are instantiated; there is no duplicate unused branch in single-task mode.
Default backbone: `tu-convnext_base`. Training without model initialization weights
starts the FPNs randomly (no implicit backbone-weight download). Particle and joint
training still load the setup-installed `localization_checkpoint.pth` by default;
semantic-only training neither reads nor requires localization weights.

The official legacy localization checkpoint contains four obsolete geometry keys:
`module.instance_mask_estimator.xym_1024`, `module.center_augmentator.xym`, and
`module.instance_center_estimator.kernel_cos` / `kernel_sin`. Only these exact keys
are ignored, and only when absent from the current runtime; ignored keys are logged.
They are coordinate caches / analytic 1D kernels, not the learned 2D localizer.
As in main's loader, a missing fixed peak-smoothing kernel leaves the runtime's
initialized value unchanged; no kernel is regenerated. Existing upstream smoothing
at peak detection remains enabled. The known first-convolution conversion
`[16,4,3,3]` → `[16,2,3,3]`
retains C/S and removes the disabled magnitude/class inputs. All other missing,
unexpected or shape-mismatched localization tensors still fail strict loading;
a partially initialized/random localizer is not silently accepted.

Version-2 checkpoints carry tasks, class order, backbone and enabled model states.
Joint checkpoints can serve either single task. Missing task weights or mismatched
class order are rejected. Particle-only inference accepts the official STEM
checkpoint through the existing shape-compatible loader. Semantic research
checkpoints are accepted only with matching class/ignore metadata and exact tensor
shapes. Particle logits are never treated as semantic weights. Checkpoints use
`weights_only=True` with a narrow NumPy metadata allowlist.

| Mode | Training labels | Inference weights |
|---|---|---|
| particles | points, including reviewed `[]` negatives | particle + localization |
| segmentation | class-ID mask and matching class metadata | semantic only |
| both | both | both branches + localization |

## Training and inference

MLflow logs losses, per-class semantic IoU/mIoU/pixel accuracy, particle
precision/recall/F1 and matched radius/localization MAE. Deterministic training and
validation diagnostics are saved at the configured interval and final epoch:
`visualizations/epoch_NNNN/{training,validation}/*-diagnostics.png`. Checkpoint:
`checkpoints/checkpoint.pth`. Host quick actions use `modelargs.emit_action`.
Particle metrics use a 20-network-pixel matching gate and the configured score
threshold; they are not physical-unit measurements. Classes absent from ground
truth and prediction are excluded from mIoU; no-match MAE is accompanied by a zero
match count.

`POST /infer` takes repeated `images` files in BF, HAADF order. Results contain
per-pair centers, scores, radii, semantic masks and task metadata. Disabled tasks
return empty particle arrays or `null` semantic results. Semantic logits resize to
original resolution before argmax. The UI supports class-mask PNG, JSON and
rasterized overlay ZIP downloads. Preannotations preserve configured particle-label ellipses and
per-class RLE brushes using actual control names.

### Inference visualization controls

The native toolbar **gear** opens live controls: particle score threshold,
minimum radius in **original-image pixels**, and named/color-coded segmentation
class visibility (including All/None). Every input immediately updates the browser
view and saves it locally; Close or Escape simply closes the panel, without rollback.
There is no confirmation step or inference/network request on settings changes.
Images and masks decode once per response; cached native-resolution canvases redraw
on the next animation frame, coalescing rapid inputs. Encoding happens only on download.

Controls and downloads for tasks excluded by the worker's explicit `tasks` metadata
are omitted. An enabled particle task with zero detections still has thresholds and
a valid empty detection download. Both BF and HAADF remain visible, with no counts or
captions; the compact legend reflects visible segmentation classes.

The browser `/infer` endpoint retains localizer candidates at score cutoff **0**,
then applies the configured `score_threshold` as the initial display threshold.
The slider can therefore reveal lower-score returned candidates without rerunning
inference; it cannot recover candidates already rejected by the model localizer.
Scores are not assumed to be probabilities: the slider range includes observed
scores above 1. Minimum radius is a client-side size filter, not a new model/NMS
parameter. CLI prediction and Label Studio preannotations retain the configured
server score cutoff. No unsupported localization/NMS knobs are exposed.

The top row contains task-appropriate downloads:

- **Download detections**: JSON with every returned candidate, unfiltered.
- **Download mask**: one ZIP containing each sample's original lossless class-ID
  PNG mask, unfiltered (class names/colors remain in the response/JSON metadata).
- **Download images**: one ZIP of native-resolution JPEGs (quality 95%) for both
  modalities of every pair, using the current display filters. JPEG is lossy;
  masks are not. Filenames are sanitized and indexed to avoid collisions.

Hidden semantic classes reveal the source image without hiding particles.
The UI uses the host's TIFF decoder when available and no new CDN dependencies.

## Installation

`setup.sh` clones published `vicoslab/CeDiRNet-STEM` **master**, applies only the
existing Python 3.11 `collections.abc` compatibility patch, and records the source
revision. All new functionality resides in this plugin. It refuses to overwrite an
existing cache. Set `CEDIRNET_STEM_DOWNLOAD_PARTICLES=0` for semantic-only setup.

The Python 3.11 environment uses NumPy **1.26.4**, one OpenCV wheel
(`opencv-python==4.11.0.86`), and the existing Torch **2.7.0 / CUDA 12.8** stack.
`requirements.txt` is resolved together with the host packages. The explicit uv
`dependency-overrides.txt` replaces the backend's moving SDK git dependency with
**label-studio-sdk 2.0.0**: newer SDK HEAD requires OpenCV 4.12 and therefore
NumPy 2, which is not this plugin's tested numerical stack. A constraint alone
cannot replace a direct git requirement. The SDK's converter and standalone
converter 0.0.59 are both supported; no headless OpenCV wheel is installed over
`cv2`. The timm **0.6.13** override replaces SMP 0.3.2's Python-3.11-incompatible
0.6.12 pin; its dependencies are resolved normally, without `--no-deps`.
`uv pip check` will still report SMP's original timm metadata pin; this is the
intentional tested override, not a missing dependency. Use the same overrides
when resolving additional packages into this environment.

The dependency solve runs from the plugin directory with relative requirements
filenames. This supports imported group paths containing spaces (for example,
`.models/ViCoS/CeDiRNet STEM/cedirnet-stem`): uv splits `--override` values on
spaces even when the shell quotes an absolute filename.

**Retrying a failed installation:** setup deliberately does not resume or erase
`$TOOLBOX_CACHE/cedirnet-stem`. Stop model workers and move the *entire* failed
model directory to a uniquely named backup, or select a new empty cache root,
then rerun setup with this plugin revision. Do not delete user checkpoints,
training output, or other models' caches. Keep the backup until the fresh install
is verified. This is a fresh-install fix, not an in-place environment migration.

Outside the Toolbox container, `CEDIRNET_STEM_MODELARGS` and
`CEDIRNET_STEM_ML_BACKEND` can name local copies of the two host packages
(normally `/opt/apps/modelargs` and `/opt/apps/label-studio-ml-backend`). They
still undergo the full dependency solve and install; they do not reuse a venv.

For testing against an existing verified environment:

```bash
TOOLBOX_CACHE=/absolute/empty/cache \
CEDIRNET_STEM_ENV=/absolute/verified/venv \
CEDIRNET_STEM_DOWNLOAD_PARTICLES=0 bash setup.sh
```

`CEDIRNET_STEM_SOURCE` can instead name an unpatched stock checkout to copy; setup
never modifies it. Environment overrides are import checks, not proof of a fresh
dependency install. Omit the overrides in the Toolbox container for normal setup.

## Code organization

Toolbox-facing files stay at the plugin root: `train.py`, `infer.py`,
`ls_adapter.py`, `setup.sh`, and the model/UI/annotation configuration. Internal
Python helpers live in the `stem_plugin/` package; tests remain in `tests/`.
The package groups dataset/annotation, model/runtime, checkpoint, serving and
visualization helpers without changing their behavior or checkpoint state keys.
Import helpers as `stem_plugin.runtime`, for example, rather than `runtime`.
Only the plugin root belongs on `PYTHONPATH`, not `stem_plugin/` itself.

## Verification

From the plugin root, add the plugin, installed upstream `src`, and current
Toolbox `apps/modelargs` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:/absolute/cache/cedirnet-stem/src:/absolute/toolbox/apps/modelargs"
# Enable released-checkpoint/default-initialization integration tests (no downloads in pytest):
export CEDIRNET_STEM_TEST_LOCALIZATION=/absolute/cache/cedirnet-stem/localization_checkpoint.pth
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES='' CEDIRNET_STEM_VERIFY_DEPENDENCIES=1 python -m pytest tests -q
node --test tests/test_ui.mjs
bash -n setup.sh
```

`tests/test_export.py` executes the **unmodified host create/export scripts**,
replacing only Label Studio network calls with SDK transport fixtures. Real RLE,
file export, manifests, loaders, one-epoch training, validation, checkpoint reload
and inference run in all three modes. Set `TOOLBOX_HOST` for a non-sibling checkout.
On Python 3.11 the harness supplies only the `Path.relative_to(walk_up=True)`
compatibility operation used by the Python-3.12 host exporter. This is not a live
Label Studio/Nexus deployment or an accuracy benchmark. Runtime tests use real
ResNet18 FPNs; separate default-ConvNeXt and released-checkpoint verification should
be retained for deployment checks.
