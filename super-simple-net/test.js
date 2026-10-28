Feature('SuperSimpleNet')

BeforeSuite(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.installSrc({
        owner: 'TestModels',
        group: 'SuperSimpleNet',
        models: ['super-simple-net'],
        url: 'https://github.com/vicoslab/toolbox-models',
        branch: 'dev'
    });
    I.section('Install model');
    I.click('SuperSimpleNet', { css: 'a.model' });
    I.click('Ok');

    I.waitForElement('a.model.installed', 30);
});

Before(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.click('SuperSimpleNet', { css: 'a.model' });
});

Scenario('Train tile demo', async ({ I }) => {
    I.section('Setup dataset');
    I.waitForURL(url => url.pathname == '/dataset');
    await I.externalDataset('../data/cell-tile-demo.zip', '/data/cell-tiles-demo/manifest.json');
    I.waitForURL(url => url.pathname.startsWith('/models/') && url.pathname.endsWith('super-simple-net'), { timeout: 5000 });

    I.training([['Batch size', 'text', '16'], ['Training duration', 'text', '2'], ['Evaluation interval', 'text', '100'], ['Flip augmentation', 'enum', 'true']], 60*5);
    const { scores } = await I.inference(() => {
        I.attachFile('.inference-tab input[type="file"]', '../data/cell-tiles-demo/damaged_0_0000_ls3_camera0.jpg');
        I.waitForElement('.activation-wrapper');
    });
    I.assert(scores[0] > 0.9, `${scores[0]} must be > 0.9`);
});

Scenario('Unsupervised', async ({ I }) => {
    I.section('Create project');
    I.click('button[onclick*="showCreateDialog"]')
    I.fillField('Name', 'My project');
    I.fillField('Path', '/data/cell-tiles-demo');
    I.click('Next');
    I.fillField('Include', 'jpg');
    I.fillField('Exclude', 'damaged');
    I.click('Create', '#manage-project');
    I.waitForURL(url => url.pathname == '/label');

    I.section('Submit annotations');
    I.click('button.close');
    I.switchTo('iframe');
    I.click('Label All Tasks');
    for (let i = 0; i < 40; i++) {
        I.click('Submit');
        I.waitForElement('button[name="submit"]');
    }
    I.switchTo(null);

    I.section('Export dataset');
    I.click('Export');
    I.waitForURL(url => url.pathname == '/export');
    I.click('Start export');
    I.waitForText('Go to training', 5);
    I.click('Go to training');

    I.training([['Batch size', 'text', '16'], ['Training duration', 'text', '20'], ['Evaluation interval', 'text', '100'], ['Flip augmentation', 'enum', 'true']], 60*5);
    const { scores } = await I.inference(() => {
        I.attachFile('.inference-tab input[type="file"]', '../data/cell-tiles-demo/damaged_0_0000_ls3_camera0.jpg');
        I.waitForElement('.activation-wrapper');
    });
    I.assert(scores[0] > 0.4, `${scores[0]} must be > 0.4`);
});

AfterSuite(async ({ I }) => {
    I.section('Cleanup');
    I.deleteModel();
    I.deleteGroup();
    I.dontSeeElement('a.model');
});
