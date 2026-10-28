Feature('CeDiRNet-STEM')

BeforeSuite(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.installSrc({
        owner: 'TestModels',
        group: 'Cedirnet-STEM',
        models: ['cedirnet-stem'],
        url: 'https://github.com/vicoslab/toolbox-models',
        branch: 'dev'
    });
    I.section('Install model');
    I.click('CeDiRNet-STEM', { css: 'a.model' });
    I.click('Ok');

    I.waitForElement('a.model.installed', 30);
});

Before(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.click('CeDiRNet-STEM', { css: 'a.model' });
});

Scenario('Nanoparticle demo', async ({ I }) => {
    I.section('Setup dataset');
    I.waitForURL(url => url.pathname == '/dataset');

    await I.externalDataset('../data/nanoparticles-demo.zip', '/data/nanoparticles/manifest.json');
    I.training([['Training duration', 'text', '15'], ['Weights save interval', 'text', '15']], 60*5);
    const results = await I.inference(() => {
        const files = ['BF', 'HAADF'].map(x => `../data/nanoparticles/PtCo_IL_a-0016_${x}.png`);
        I.usePlaywrightTo('Attach multiple files', async ({ page }) => {
            await page.locator('.inference-tab input[type="file"]').setInputFiles(files);
        });
        I.waitForElement('.stem-results', 5);
    });
    const centers = [[0.5, 0.5]];
    const holds = centers.every(([cx, cy]) => results.centers[0].some(([px, py]) => Math.hypot(cx-px, cy-py) < 0.05));
    I.assert(holds, 'Predictions didn\'t contain center at expected position(s)');
});

AfterSuite(async ({ I }) => {
    I.section('Cleanup');
    I.deleteModel();
    I.deleteGroup();
    I.dontSeeElement('a.model');
});
