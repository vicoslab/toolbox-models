# Host-layout rendering regression

Install Playwright, Pillow and NumPy in a test environment (`playwright install chromium`
if no Chromium is installed). Run from any directory, using absolute script paths:

```sh
export TOOLBOX_HOST=/path/to/toolbox
python /path/to/cedirnet-stem/tests/browser_host_fixture.py
# In another terminal:
export STEM_BROWSER_ARTIFACTS=/path/to/evidence
python /path/to/cedirnet-stem/tests/verify_browser_host.py
```

Optional: `STEM_BROWSER_EXECUTABLE` selects an installed Chromium executable;
`STEM_BROWSER_PORT` selects the port (default 18768). `STEM_REAL_PNG_DIR` additionally
checks `PtCo_IL_a-0016_BF.png` and `PtCo_IL_a-0016_HAADF.png` from a real dataset.
Without that variable only the explicitly synthetic image fixtures are exercised.

The fixture reads the **actual host worker form from templates/model.html**, retains
its model-overview/inference/details/fieldset containment, loads unmodified host CSS
and model.js after plugin markup, and uploads Files through infer-image's shadow file
input. The real host ElementInternals/FormData, makeRequest POST, infer event, callback
and replaceChildren flow run in Chromium. Only the backend predictions are synthetic;
this is a rendering test, not a model-accuracy test or an authenticated deployment test.
Loading/toast plumbing is stubbed, not model.js or the plugin callback.

The previous free-standing-form harness missed fieldset's `auto 1fr` grid. Absolute
slotted content has no intrinsic width: plugin images decoded successfully but had
zero client width/height. Require both natural and layout dimensions, viewport
screenshots and independent ZIP/JPEG (lossy tolerance) and class-ID PNG (exact) pixel
checks. Synchronize on completed animation frames, not only particle count: class
changes can retain the same count. The regression instruments requests and image
`src`/canvas `toBlob` calls to prove that display filtering neither reruns inference
nor decodes or encodes images. A burst of inputs must draw the latest state exactly
once in the next animation frame, including a changed overlay pixel; measured
latency is recorded, not presented as a hardware-independent performance guarantee.

It also covers the exact unified top download labels, all-sample archives, legend
palette/visibility, no captions/status, task-specific control/download omission,
enabled-but-empty particle results, immediate persistence, Close/Escape without
rollback, corrupt storage, reset, and real BF/HAADF source pixels when requested.
