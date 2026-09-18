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
screenshots and independent PNG/ZIP pixel checks. Synchronize on render completion,
not only particle count: class changes can retain the same count during async rendering.
The regression also covers legend palette/visibility, no inline captions/status,
threshold and radius filters, class controls, task modes, persistence and reset.
