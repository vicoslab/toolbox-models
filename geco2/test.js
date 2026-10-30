Feature('GeCo2')

BeforeSuite(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.installSrc({
        owner: 'TestModels',
        group: 'GeCo2',
        models: ['geco2'],
        url: 'https://github.com/vicoslab/toolbox-models',
        branch: 'dev'
    });
    I.section('Install model');
    I.click('GeCo2', { css: 'a.model' });
    I.click('Ok');

    I.waitForElement('a.model.installed', 30);
});

Before(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.click('GeCo2', { css: 'a.model' });
    I.waitForURL(url => url.pathname.startsWith('/models/') && url.pathname.endsWith('geco2'), { timeout: 5000 });
});

function matches([rx, ry, rw, rh], [px, py, pw, ph]) {
    return Math.abs(rx - px) + Math.abs(ry - py) + Math.abs(rw - pw) + Math.abs(rh - ph) < 0.05;
}

Scenario('Nanoparticle', async ({ I }) => {
    let ref;
    let boxes = [];
    const preds = await I.inference(() => {
        I.attachFile('.inference-tab input[type="file"]', '../data/nanoparticles/PtCo_IL_a-0016_BF.png');
        I.waitForElement('.showing');

        I.usePlaywrightTo('Select exemplar object', async ({ page }) => {
            const area = page.locator('.area');
            const ref = await area.boundingBox();
            await area.dragTo(area, { sourcePosition: { x: 0.48*ref.width, y: 0.48*ref.height }, targetPosition: { x: 0.54*ref.width, y: 0.54*ref.height }});
            // set threshold to known value
            await page.evaluate('localStorage["toolbox-ui-detection-dialog"] = JSON.stringify({ hideBoxes: false, hideMasks: false, hideExemplars: false, threshold: 0.42 })');
        });
        I.click('Submit');
        I.waitForElement('#results .box', 5);
        I.grabElementBoundingRect('#results .reference').then(x => { ref = x; });
        I.grabWebElements('#results .box').then(x => Promise.all(x.map(el => el.getBoundingBox()))).then(x => boxes.push(...x));
        I.wait(0.25);
    });
    // preds.boxes = preds.boxes.map(([x,y,width,height]) => [x*ref.width, y*ref.width, width*ref.width, height*ref.height]);
    boxes = boxes.map(({ x, y, width, height }) => [ (x - ref.x )/ref.width, (y - ref.y)/ref.height, width/ref.width, height/ref.height]);
    const holds = preds.scores.every((x,i) => x < 0.42 || boxes.some(dims => matches(preds.boxes[i], dims)));
    I.assert(holds, 'Displayed predictions must match calculated positions');
});

AfterSuite(async ({ I }) => {
    I.section('Cleanup');
    I.deleteModel();
    I.deleteGroup();
    I.dontSeeElement('a.model');
});
