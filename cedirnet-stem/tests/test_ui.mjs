import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = fs.readFileSync(new URL("../ui.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "ui.html must contain a script");
const helpers = script.split('document.getElementById("infer").onInference')[0];
vm.runInThisContext(`${helpers}\nglobalThis.__cedirnetUi = { buildZip, crc32, assertClassicZipLimit };`);
const { buildZip, crc32, assertClassicZipLimit } = globalThis.__cedirnetUi;

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
