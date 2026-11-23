import * as fs from 'node:fs';

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

    I.waitForElement('.toast.info', 5);
    I.waitForInvisible('.toast.info', 60 * 5);
    I.waitForElement('a.model.installed');
});

Before(({ I }) => {
    I.amOnPage('/models?tour=1');
    I.click('CeDiRNet-STEM', { css: 'a.model' });
});

Scenario('Nanoparticle demo', async ({ I }) => {
    I.section('Setup dataset');
    I.waitForURL(url => url.pathname.startsWith('/dataset'));

    await I.externalDataset('../data/nanoparticles-demo.zip', '/data/nanoparticles/manifest.json');
    I.training([['Training duration', 'text', '15'], ['Weights save interval', 'text', '15']], 60 * 5);
    const results = await I.inference(() => {
        const files = ['BF', 'HAADF'].map(x => `../data/nanoparticles/PtCo_IL_a-0016_${x}.png`);
        I.usePlaywrightTo('Attach multiple files', async ({ page }) => {
            await page.locator('.inference-tab input[type="file"]').setInputFiles(files);
        });
        I.waitForElement('.stem-results', 5);
    });
    const centers = [[0.5, 0.5]];
    const holds = centers.every(([cx, cy]) => results.centers[0].some(([px, py]) => Math.hypot(cx - px, cy - py) < 0.05));
    I.assert(holds, 'Predictions didn\'t contain center at expected position(s)');
});

Scenario('Default weights with single image', async ({ I }) => {
    I.click('Inference');
    I.waitForURL(url => url.pathname.startsWith('/model'));
    const results = await I.inference(() => {
        const files = [`../data/nanoparticles/PtCo_IL_a-0016_BF.png`];
        I.usePlaywrightTo('Attach single file', async ({ page }) => {
            await page.locator('.inference-tab input[type="file"]').setInputFiles(files);
        });
        I.click('BF');
        I.click('Run Inference');
        I.waitForElement('.stem-results', 5);
    });
    const centers = [[0.5, 0.5]];
    const holds = centers.every(([cx, cy]) => results.centers[0].some(([px, py]) => Math.hypot(cx - px, cy - py) < 0.05));
    I.assert(holds, 'Predictions didn\'t contain center at expected position(s)');
});

Scenario('Smart tool and semantic labels', async ({ I }) => {
    I.click('Inference');
    I.waitForURL(url => url.pathname.startsWith('/model'));

    await I.worker(() => {
        I.click('Dataset');

        I.section('Create project');
        I.click('button[onclick*="showCreateDialog"]')
        I.fillField('Name', 'Cedirnet Seg');
        I.fillField('Path', '/data/nanoparticles-' + Date.now());
        const root = '../data/nanoparticles/';
        const files = fs.readdirSync(root).map(x => root + x);
        I.usePlaywrightTo('Attach multiple files', async ({ page }) => {
            await page.locator('#dataset-upload').setInputFiles(files);
        });
        I.fillField('Include', 'png');
        I.click('Grouped image samples');
        I.fillField('Group size', 2);
        I.click('Shared');
        I.click('Create', '#manage-project');
        I.waitForURL(url => url.pathname.startsWith('/label'));

        I.section('Submit annotations');
        I.click('button.close');
        I.switchTo('iframe');
        I.click('Label All Tasks');
        const annotations = [
            {
                Carbon: [[0.198, 0.807], [0.198, 0.587], [0.299, 0.564], [0.380, 0.414], [0.335, 0.331], [0.352, 0.25], [0.440, 0.226], [0.508, 0.262], [0.608, 0.264], [0.696, 0.326], [0.700, 0.459], [0.651, 0.596], [0.664, 0.728], [0.760, 0.752], [0.788, 0.799], [0.801, 0.818]],
                Film: [],
                Vacuum: [[0.205, 0.555], [0.292, 0.544], [0.365, 0.418], [0.322, 0.316], [0.344, 0.245], [0.429, 0.217], [0.510, 0.243], [0.626, 0.260], [0.711, 0.324], [0.711, 0.482], [0.662, 0.596], [0.677, 0.698], [0.771, 0.739], [0.799, 0.782], [0.797, 0.209], [0.205, 0.213]],
            },
            {
                Carbon: [[0.252, 0.767], [0.254, 0.604], [0.344, 0.497], [0.373, 0.416], [0.444, 0.378], [0.521, 0.414], [0.564, 0.478], [0.553, 0.474], [0.649, 0.446], [0.649, 0.348], [0.675, 0.316], [0.756, 0.282], [0.75, 0.773]],
                Film: [[0.388, 0.388], [0.412, 0.303], [0.491, 0.279], [0.747, 0.269], [0.643, 0.322], [0.638, 0.429], [0.568, 0.448], [0.512, 0.386], [0.446, 0.363]],
                Vacuum: [[0.247, 0.585], [0.247, 0.271], [0.474, 0.269], [0.399, 0.299], [0.367, 0.399], [0.326, 0.491]]
            },
            {
                Carbon: [[0.0876, 0.999], [0.0406, 0.925], [0.00855, 0.684], [0.126, 0.551], [0.145, 0.412], [0.348, 0.259], [0.491, 0.241], [0.641, 0.139], [0.748, 0.120], [0.827, 0.177], [0.944, 0.194], [0.998, 0.250], [0.996, 0.999]],
                Film: [[0.340, 0.239], [0.355, 0.120], [0.436, 0.0214], [0.810, 0.0214], [0.938, 0.162], [0.818, 0.160], [0.741, 0.0983], [0.641, 0.118], [0.491, 0.220]],
                Vacuum: [[0.004, 0.671], [0.115, 0.545], [0.135, 0.400], [0.331, 0.252], [0.344, 0.113], [0.406, 0.0299], [0.00641, 0.0214]],
            },
            {
                Carbon: [[0.00427, 0.999], [0.0726, 0.876], [0.0534, 0.637], [0.105, 0.389], [0.00641, 0.440], [0.00, 0.0214], [0.0684, 0.0192], [0.177, 0.167], [0.139, 0.314], [0.267, 0.186], [0.468, 0.179], [0.677, 0.312], [0.848, 0.498], [0.878, 0.746], [0.806, 0.865], [0.979, 0.906], [0.991, 0.999]],
                Film: [[0.299, 0.167], [0.468, 0.100], [0.650, 0.203], [0.688, 0.280], [0.895, 0.444], [0.964, 0.658], [0.989, 0.874], [0.829, 0.846], [0.891, 0.744], [0.865, 0.496], [0.679, 0.303], [0.472, 0.171]],
                Vacuum: [[0.156, 0.269], [0.194, 0.143], [0.120, 0.0299], [0.991, 0.0299], [0.991, 0.735], [0.910, 0.425], [0.699, 0.278], [0.654, 0.179], [0.470, 0.0791], [0.286, 0.145]]
            },
            {
                Carbon: [[0.872, 0.891], [0.714, 0.893], [0.530, 0.726], [0.361, 0.812], [0.207, 0.746], [0.135, 0.624], [0.128, 0.487], [0.286, 0.295], [0.481, 0.226], [0.615, 0.256], [0.726, 0.412], [0.816, 0.190], [0.801, 0.150], [0.868, 0.141]],
                Film: [[0.278, 0.271], [0.500, 0.173], [0.690, 0.252], [0.746, 0.145], [0.797, 0.152], [0.799, 0.190], [0.731, 0.370], [0.626, 0.248], [0.491, 0.216]],
                Vacuum: [[0.699, 0.895], [0.124, 0.889], [0.122, 0.643], [0.192, 0.746], [0.346, 0.821], [0.513, 0.756]]
            }
        ];
        I.waitForElement('button[aria-label="polygon-tool"]', 5); // initial load might take a bit longer
        for (const ann of annotations) {
            I.waitForElement('button[aria-label="polygon-tool"]');
            I.seeElement('.lsf-tree__node_type_ellipseregion');
            I.click('button[aria-label="Hide all regions"]');
            I.click('button[aria-label="polygon-tool"]');
            I.usePlaywrightTo('Select exemplar object', async ({ page }) => {
                const locator = page.frameLocator('iframe');
                const area = locator.locator('.lsf-image');
                const ref = await area.boundingBox();
                for (const [k, v] of Object.entries(ann)) {
                    if (v.length == 0) continue;
                    v.push(v[0]);
                    const label = locator.locator('.lsf-label', { hasText: k });
                    await label.click();
                    await page.evaluate('document.querySelector("iframe").contentWindow.document.querySelector(".lsf-main-content").scrollTop = 0');
                    for (const [x, y] of v) {
                        await new Promise(r => setTimeout(r, 10));
                        await area.locator('canvas').click({
                            position: { x: x * ref.width, y: y * ref.height },
                            scroll: 'none',
                            force: true
                        });
                    }
                }
            });
            I.click('Submit');
        }
    });

    I.section('Export dataset');
    I.switchTo();
    I.click('Export');
    I.waitForURL(url => url.pathname.startsWith('/export'));
    I.click('Start export');
    I.waitForText('Go to training', 5);
    I.click('Go to training');

    I.training([['Training duration', 'text', '40'], ['Weights save interval', 'text', '40'], ['Enable multiclass segmentation', 'enum', 'true'], ['Batch size', 'text', '1']], 60 * 5);
    const results = await I.inference(() => {
        const files = ['BF', 'HAADF'].map(x => `../data/nanoparticles/PtCo_IL_a-0016_${x}.png`);
        I.usePlaywrightTo('Attach multiple files', async ({ page }) => {
            await page.locator('.inference-tab input[type="file"]').setInputFiles(files);
        });
        I.waitForElement('.stem-results', 5);
    });
    I.assert(results.segmentation[0] !== null)
});

AfterSuite(async ({ I }) => {
    I.section('Cleanup');
    I.deleteModel();
    I.deleteGroup();
    I.dontSeeElement('a.model');
});
