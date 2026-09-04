import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = fs.readFileSync(new URL("../ui.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "ui.html must contain a script");
const helpers = script.split('document.getElementById("infer").onInference')[0];
globalThis.document = { getElementById: () => ({}) };
vm.runInThisContext(`${helpers}\nglobalThis.__cedirnetUi = { filterDetectionIndices, thresholdSliderMaximum };`);
const { filterDetectionIndices, thresholdSliderMaximum } = globalThis.__cedirnetUi;

test("threshold filter updates selected detections without backend inference", () => {
    const scores = [0.1, 0.5, 0.5001, 0.9];
    assert.deepEqual(filterDetectionIndices(scores, 0.5), [1, 2, 3]);
    assert.deepEqual(filterDetectionIndices(scores, 0.75), [3]);
    assert.deepEqual(filterDetectionIndices(scores, 1.0), []);
});

test("threshold slider supports model scores above one", () => {
    assert.equal(thresholdSliderMaximum([[0.2, 0.9], [1.7]]), 1.7);
    assert.equal(thresholdSliderMaximum([[]]), 1.0);
});

test("UI exposes a realtime confidence threshold slider", () => {
    assert.match(html, /thresholdInput\.type = "range"/);
    assert.match(html, /thresholdInput\.addEventListener\("input"/);
    assert.match(html, /Confidence threshold/);
});

test("inference callback rerenders markers immediately when threshold changes", () => {
    class Element {
        constructor(tag) {
            this.tag = tag;
            this.children = [];
            this.style = {};
            this.dataset = {};
            this.listeners = {};
            this.classes = new Set();
            this.classList = { add: name => this.classes.add(name) };
            this.clientWidth = 640;
            this.clientHeight = 400;
        }
        append(...children) { children.forEach(child => this.appendChild(child)); }
        appendChild(child) { child.parent = this; this.children.push(child); return child; }
        addEventListener(name, callback) { this.listeners[name] = callback; }
        dispatchEvent(event) { this.listeners[event.type]?.(event); }
        querySelectorAll(selector) {
            const name = selector.replace(/^\./, "");
            return this.children.flatMap(child => [
                ...(child.classes?.has(name) ? [child] : []),
                ...(child.querySelectorAll?.(selector) || []),
            ]);
        }
        remove() {
            if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
        }
    }

    const templateElement = { content: { children: [new Element("svg")] } };
    const inferElement = {};
    const fakeDocument = {
        getElementById: id => id === "template-marker" ? templateElement : inferElement,
        createElement: tag => new Element(tag),
        importNode: () => new Element("svg"),
    };
    const context = vm.createContext({
        document: fakeDocument,
        URL: { createObjectURL: () => "blob:test", revokeObjectURL: () => {} },
        Number,
        Math,
    });
    vm.runInContext(script, context);
    const [controls, group] = inferElement.onInference(
        { getAll: () => [{ name: "test.png" }] },
        {
            centers: [[[0.2, 0.2, 1], [0.5, 0.5, 1], [0.8, 0.8, 1]]],
            scores: [[0.2, 0.6, 0.9]],
            angles: [[0, 45, 90]],
        },
    );
    const image = group.children[0].children[0];
    image.onload();
    assert.equal(group.querySelectorAll(".cedirnet-marker").length, 2);

    const slider = controls.children[1];
    slider.value = "0.8";
    slider.dispatchEvent({ type: "input" });
    assert.equal(group.querySelectorAll(".cedirnet-marker").length, 1);
    assert.equal(controls.children[2].value, "0.80");
});
