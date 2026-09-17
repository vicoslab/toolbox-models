# Label Studio editor fixtures

These JSON files are **outputs of the real Label Studio 1.23.0 browser editor**
(`annotation.serializeAnnotation()`), using the distributed JS bundles extracted
from `label_studio-1.23.0-py3-none-any.whl`, not SDK transport mocks.
The isolated editor used two synthetic 64×64 grayscale images in a shared Image
`valueList`. No user dataset or annotation server was modified.

- `ls-1.23-current.json`: current plugin `config.yml`; SDK brush RLE inputs for
  Carbon, Film, Vacuum, Ignore and an aliased nanoparticle ellipse were loaded via
  `deserializeAnnotation`, then serialized by the real region models. Thus this
  is an import/serialization fixture, not a claim of manual drawing of each brush.
- `ls-1.23-magicwand.json`: the above annotation plus an actual Magicwand click on
  the constant image, followed by assigning Carbon to the selected region.
  The editor emits **two results with the same ID**: `magicwand` with unlabeled
  RLE and `brushlabels` containing identical RLE plus the selected class.
  Before assigning a class it emitted only unlabeled `magicwand`; this must not
  be silently discarded or interpreted as a negative/background.
- `ls-1.23-magicwand-classes.json`: snapshots of that actual wand region after
  selecting each of Carbon, Film, Vacuum, Ignore. Each entry contains its paired
  results; the complete image is selected in every snapshot.
- `ls-1.23-separate.json`: a temporary editor config using separate `Labels
  name="semantic"` and `Ellipse`, `Rectangle`, `Polygon` controls. Seed geometry
  was imported then serialized by the editor. Labels results repeat geometry;
  joining is by region ID, target and `item_index`, never list position.

The alternate config was only used in the isolated schema check. The plugin's
chosen sidebar UI is unchanged. The user-reported failing export was unavailable;
these fixtures establish real format coverage, not its proven provenance.
