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
2. Select **Nanoparticles (N)**, **Segmentation (S)**, or both at the top. These
   checkboxes show independent collapsible annotation sections, not training flags
   or review confirmations. Collapsing/hiding a section does not delete its regions.
   Draw particle **ellipses** (Particle: **P**). Paint semantic regions using
   **Carbon (1)**, **Film (2)**, **Vacuum (3)**, **Ignore (4)**. The image view shares
   regions across the aligned pair.
3. For a reviewed image with no particles, explicitly check **No nanoparticles**
   (Alt+N). Do not check it if particle regions exist; export rejects that conflict.
   Submit one annotation per pair. Resolve multiple annotators before export;
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

The annotation template uses documented `Choices showInline`, `Choice/Label hotkey`,
`View visibleWhen="choice-selected" whenTagName/whenChoiceValue`, and
`Collapse/Panel open="true"` syntax. No custom JavaScript or host changes are needed.

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
Default backbone: `tu-convnext_base`. Training without initialization weights starts
from random weights (no implicit pretrained-weight download).

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
rasterized overlay ZIP downloads. Preannotations preserve Particle ellipses and
per-class RLE brushes using actual control names.

## Installation

`setup.sh` clones published `vicoslab/CeDiRNet-STEM` **master**, applies only the
existing Python 3.11 `collections.abc` compatibility patch, and records the source
revision. All new functionality resides in this plugin. It refuses to overwrite an
existing cache. Set `CEDIRNET_STEM_DOWNLOAD_PARTICLES=0` for semantic-only setup.

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
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
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
