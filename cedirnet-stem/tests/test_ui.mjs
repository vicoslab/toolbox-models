import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = fs.readFileSync(new URL("../ui.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "ui.html must contain a script");
const helpers = script.split('document.getElementById("@ALIAS@").onInference')[0];
vm.runInThisContext(`${helpers}\nglobalThis.__cedirnetUi = { buildZip, crc32, assertClassicZipLimit, sampleResult };`);
const { buildZip, crc32, assertClassicZipLimit, sampleResult } = globalThis.__cedirnetUi;

test('view filters reject bad geometry and apply score and original-pixel radius thresholds', () => {
  assert.equal(typeof filterParticles, 'function');
  const result = {centers:[[.2,.3],[.5,.6],[.1,.1],[NaN,.3]], scores:[.2,2,.9,3], radii:[4,10,2,6]};
  assert.deepEqual(filterParticles(result, {threshold:.5,minRadius:3}).scores, [2]);
  assert.equal(filterParticles(result, {threshold:0,minRadius:0}).scores.length, 3);
});

test('view normalization bounds numeric controls and handles corrupt saved values', () => {
  assert.equal(typeof normalizeView, 'function');
  assert.deepEqual(normalizeView({threshold:Infinity,minRadius:-5,hiddenClasses:['Film',3]}, 2, .5),
    {threshold:.5,minRadius:0,hiddenClasses:['Film']});
  assert.equal(normalizeView({threshold:50}, 2).threshold,2);
  assert.equal(normalizeView({threshold:null}, 2).threshold,.5);
});

test('semantic class visibility changes only alpha for matching class IDs', () => {
  assert.equal(typeof semanticPixels, 'function');
  const pixels = semanticPixels(new Uint8ClampedArray([0,0,0,255,1,1,1,255]),
    {classes:['Carbon','Film'],colors:[[255,0,0],[0,255,0]]}, ['Film']);
  assert.deepEqual(Array.from(pixels),[255,0,0,120,0,255,0,0]);
});

test('semantic recoloring reuses native pixel storage and clears unknown IDs', () => {
  const output = new Uint8ClampedArray([255,255,255,255,255,255,255,255]);
  assert.equal(semanticPixels(new Uint8ClampedArray([0,0,0,255,3,3,3,255]),
    {classes:['Carbon'],colors:[[255,0,0]]}, [], output), output);
  assert.deepEqual(Array.from(output), [255,0,0,120,0,0,0,0]);
});

test('download colors are opaque for every configured ID, including zero; ignore is neutral', () => {
  const colors = Array.from({length:255}, (_, id) => [id, 255-id, 73]);
  const ids = new Uint8ClampedArray(Array.from({length:256}, (_, id) => [id,id,id,255]).flat());
  const original = ids.slice();
  const pixels = colorMaskPixels(ids, {colors});
  for (let id=0; id<255; id++) assert.deepEqual(Array.from(pixels.slice(id*4,id*4+4)), [...colors[id],255]);
  assert.deepEqual(Array.from(pixels.slice(255*4)), [128,128,128,255]);
  assert.deepEqual(ids, original);
});

test('all independent task result modes preserve masks without particle output', () => {
  const segmentation = { classes: ['Carbon', 'Film', 'Vacuum'], mask_png: 'data:image/png;base64,AA==' };
  assert.deepEqual(sampleResult({segmentation:[segmentation]}, 0), {centers:[],scores:[],radii:[],segmentation});
  assert.equal(sampleResult({centers:[[[.5,.5]]],scores:[[.9]],radii:[[4]]},0).segmentation,null);
  assert.equal(sampleResult({segmentation:[segmentation],centers:[[[.5,.5]]]},0).centers.length,1);
});

test('explicit task metadata controls availability even with no detections', () => {
  assert.deepEqual(enabledTasks({tasks:{nanoparticles:true,segmentation:false},scores:[[]]}),
    {nanoparticles:true,segmentation:false});
  assert.deepEqual(enabledTasks({tasks:{nanoparticles:false,segmentation:true}}),
    {nanoparticles:false,segmentation:true});
});

test('archive filenames are safe, indexed and JPEG', () => {
  assert.equal(annotatedFilename({name:'../../a b.png'},0),'001-..-..-a-b-cedirnet.jpg');
  assert.notEqual(annotatedFilename({name:'../../a b.png'},0), annotatedFilename({name:'../../a b.png'},1));
});

function uint32(bytes, offset) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true);
}

test("CRC-32 matches the standard check vector", () => {
  assert.equal(crc32(new TextEncoder().encode("123456789")), 0xcbf43926);
});

test("ZIP contains valid local, central, and end records for UTF-8 names", async () => {
  const zip = new Uint8Array(await (await buildZip([
    { name: "particle-µ.png", blob: new Blob([new Uint8Array([1, 2, 3])]) },
  ])).arrayBuffer());
  assert.equal(uint32(zip, 0), 0x04034b50);
  const centralOffset = zip.findIndex((_, index) => uint32(zip, index) === 0x02014b50);
  assert.ok(centralOffset > 0);
  const endOffset = zip.length - 22;
  assert.equal(uint32(zip, endOffset), 0x06054b50);
  assert.equal(uint32(zip, endOffset + 16), centralOffset);
});

test("empty ZIP is valid", async () => {
  const zip = new Uint8Array(await (await buildZip([])).arrayBuffer());
  assert.equal(zip.length, 22);
  assert.equal(uint32(zip, 0), 0x06054b50);
});

test("classic ZIP rejects values outside 16-bit and 32-bit fields", () => {
  assert.throws(() => assertClassicZipLimit(0x10000, 16, "entry count"), /entry count/);
  assert.throws(() => assertClassicZipLimit(0x100000000, 32, "archive size"), /archive size/);
});

test("ZIP rejects too many entries before processing them", async () => {
  await assert.rejects(buildZip(new Array(0x10000).fill(null)), /entry count/);
});

test("ZIP rejects an oversized UTF-8 filename", async () => {
  await assert.rejects(
    buildZip([{ name: "x".repeat(0x10000), blob: new Blob([]) }]),
    /filename length/,
  );
});
