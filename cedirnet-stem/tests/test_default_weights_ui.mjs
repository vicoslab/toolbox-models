// Exercise Toolbox's actual required-option validator with the plugin schema.
// Usage: node tests/test_default_weights_ui.mjs /path/to/toolbox
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const toolbox = process.argv[2];
assert.ok(toolbox, 'Pass the Toolbox source directory');
const schema = JSON.parse(fs.readFileSync(path.join(root, 'model.json'), 'utf8'));
const template = fs.readFileSync(path.join(toolbox, 'apps/nexus/templates/model.html'), 'utf8');
const source = template.match(/    function checkRequiredOptions\(\) \{[\s\S]*?\n    \}/)?.[0];
assert.ok(source, 'Find the real Toolbox validator');
function validate(values, stage = 'infer') {
    const errors = [];
    const context = vm.createContext({values, stage, options: schema.properties,
        showToast: (kind, text) => errors.push([kind, text])});
    vm.runInContext(source, context);
    return {ok: vm.runInContext('checkRequiredOptions()', context), errors};
}
assert.equal(validate({}).ok, true, 'Fresh particle worker must accept downloaded default weights');
assert.equal(validate({model: 'default', localisation: 'default'}).ok, true);
assert.equal(validate({model: '/tmp/custom.pth'}).ok, true);
assert.equal(validate({}, 'train').ok, false, 'Training manifest remains required');
assert.equal(validate({manifest: '/tmp/manifest.json'}, 'train').ok, true);
assert.equal(schema.properties.model.default, 'default');
assert.equal(schema.properties.localisation.default, 'default');
console.log('PASS: real Toolbox required-option validator accepts default/custom inference weights; training manifest remains required');
