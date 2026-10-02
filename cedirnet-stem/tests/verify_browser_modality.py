"""Singleton upload checks using unmodified host components and synthetic predictions."""
from pathlib import Path
import io
import json
import os
import tempfile
import zipfile
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(os.environ.get('STEM_BROWSER_ARTIFACTS', tempfile.mkdtemp(prefix='stem-modality-browser-')))
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:' + os.environ.get('STEM_BROWSER_PORT', '18768')
report = {'fixture': 'Real host upload/FormData and plugin UI; synthetic predictions', 'checks': []}

def check(name, value):
    assert value, name
    report['checks'].append(name)

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=os.environ.get('STEM_BROWSER_EXECUTABLE'),
        headless=True, args=['--no-sandbox', '--disable-gpu'])
    page = browser.new_page(viewport={'width': 1280, 'height': 960}, accept_downloads=True)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(URL + '/')
    def ready(count=None, label=None):
        page.wait_for_function("document.querySelector('.stem-results')?.dataset.revision && document.querySelector('.stem-results').getAttribute('aria-busy')==='false'")
        if count is not None:
            page.wait_for_function('(n)=>document.querySelectorAll(".stem-annotated").length===n', arg=count)
        if label is not None:
            page.wait_for_function('(label)=>document.querySelector(".stem-annotated")?.getAttribute("aria-label")===label', arg=label)
        page.wait_for_function("document.querySelector('.stem-results')?.getAttribute('aria-busy')==='false'")
    def log():
        return page.request.get(URL + '/request-log.json').json()
    def reset():
        page.evaluate("document.querySelector('form').reset()")
    def upload(names):
        page.locator('infer-image input[type=file]').set_input_files([
            {'name': name, 'mimeType': 'image/png', 'buffer': page.request.get(URL + '/0-BF.png').body()}
            for name in names])
    def download(label, filename):
        with page.expect_download() as pending:
            page.get_by_role('button', name=label, exact=True).click()
        pending.value.save_as(OUT / filename)
        return OUT / filename
    ready()
    initial = len(log())
    reset()
    upload(['generic.png'])
    page.wait_for_timeout(150)
    check('singleton upload cannot submit before modality choice', len(log()) == initial)
    dialog = page.get_by_role('dialog', name='STEM image modality')
    check('singleton upload asks BF or HAADF', dialog.is_visible())
    run = dialog.get_by_role('button', name='Run inference', exact=True)
    check('modality has no guessed or remembered default', run.is_disabled())
    dialog.get_by_role('button', name='Cancel', exact=True).click()
    check('Cancel sends no request', len(log()) == initial)
    check('Cancel clears pending upload', page.locator('infer-image input[type=file]').evaluate('(e)=>e.files.length') == 0)
    upload(['generic.png'])
    dialog.wait_for(state='visible')
    page.keyboard.press('Escape')
    check('Escape cancels without inference', len(log()) == initial and not dialog.is_visible())
    for modality in ['BF', 'HAADF']:
        reset()
        before = len(log())
        upload(['generic.png'])
        dialog.wait_for(state='visible')
        check(modality + ': every singleton upload asks again', run.is_disabled())
        dialog.get_by_role('radio', name=modality, exact=True).check()
        check(modality + ': choice enables inference', not run.is_disabled())
        run.click()
        ready(1, modality + ' inference overlay')
        sent = log()
        check(modality + ': exactly one real multipart request', len(sent) == before + 1)
        check(modality + ': explicit modality and only original uploaded image',
            sent[-1]['modality'] == modality and sent[-1]['images'] == ['generic.png'])
        canvases = page.locator('.stem-annotated')
        check(modality + ': only provided modality is rendered', canvases.count() == 1 and
            canvases.first.get_attribute('aria-label') == modality + ' inference overlay')
        check(modality + ': native and visible dimensions', canvases.first.evaluate('(c)=>[c.width,c.height,c.clientWidth,c.clientHeight]') == [800, 400, 512, 256])
        detections = json.loads(download('Download detections', modality + '-detections.json').read_text())
        check(modality + ': download identifies missing detector', len(detections) == 1 and
            detections[0]['image-' + modality] == 'generic.png' and
            detections[0]['image-' + ('HAADF' if modality == 'BF' else 'BF')] is None)
        archive = zipfile.ZipFile(download('Download visualizations', modality + '-visualizations.zip'))
        check(modality + ': JPEG ZIP contains one native-resolution overlay', len(archive.namelist()) == 1 and
            Image.open(io.BytesIO(archive.read(archive.namelist()[0]))).size == (800, 400))
        masks = zipfile.ZipFile(download('Download mask', modality + '-masks.zip'))
        mapping = json.loads(masks.read('classes.json'))
        check(modality + ': raw/color mask ZIP contains one sample', len(mapping['samples']) == 1 and len(masks.namelist()) == 3)
        for key in ['class_id', 'color']:
            check(modality + ': ' + key + ' mask stays native-resolution',
                Image.open(io.BytesIO(masks.read(mapping['samples'][0][key]))).size == (800, 400))
    reset()
    before = len(log())
    upload(['a-BF.png', 'a-HAADF.png'])
    ready(2)
    check('paired upload bypasses singleton prompt', not dialog.is_visible() and len(log()) == before + 1)
    check('paired upload clears stale singleton modality', log()[-1]['modality'] == 'paired')
    check('paired rendering still contains BF and HAADF', page.locator('.stem-annotated').evaluate_all('els=>els.map(e=>e.getAttribute("aria-label"))') == ['BF inference overlay', 'HAADF inference overlay'])
    reset()
    before = len(log())
    page.evaluate("window.notices=[]; window.showToast=(level,error)=>notices.push(String(error))")
    upload(['a.png', 'b.png', 'c.png'])
    page.wait_for_timeout(150)
    check('odd multi-image uploads are rejected before inference', len(log()) == before and bool(page.evaluate('notices.length')))
    check('no JavaScript exceptions', errors == [])
    browser.close()
report['check_count'] = len(report['checks'])
(OUT / 'browser-modality-report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
