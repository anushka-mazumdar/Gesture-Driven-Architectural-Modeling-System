import test from 'node:test';
import assert from 'node:assert/strict';
import { createCandidatePanelController } from '../webview_app/web/candidate_panel.mjs';

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.textContent = '';
    this.hidden = false;
    this.className = '';
    this.attributes = {};
    this.listeners = {};
    this.dataset = {};
    this.style = { setProperty(name, value) { this[name] = value; } };
    this.disabled = false;
    this.rect = { left: 0, top: 0, right: 100, bottom: 100 };
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  click() { this.listeners.click?.(); }
  getAttribute(name) { return this.attributes[name]; }
  getBoundingClientRect() { return this.rect; }
  append(...elements) { this.children.push(...elements); }
  appendChild(element) { this.children.push(element); return element; }
  replaceChildren(...elements) { this.children = elements; }
  get allText() { return this.textContent + this.children.map((item) => item.allText).join(''); }
}

function panelFixture(options) {
  const doc = { createElement(tag) { return new FakeElement(tag, doc); } };
  const root = new FakeElement('aside', doc);
  return { root, controller: createCandidatePanelController(root, options) };
}

test('renders horizontal cards in API order and highlights selected ID', () => {
  const { root, controller } = panelFixture();
  controller.update({
    status: 'ok', display_label: 'Circle',
    selected_id: 'cylinder', candidates: [
      { id: 'sphere', label: 'Sphere' },
      { id: 'cylinder', label: 'Cylinder' },
    ],
  });
  assert.equal(root.hidden, false);
  const track = root.children[1].children[0];
  assert.equal(track.getAttribute('aria-orientation'), 'horizontal');
  assert.equal(track.style.transform, 'translateX(-100%)');
  assert.deepEqual(track.children.map((card) => [
    card.children[0].textContent, card.children[1].textContent,
    card.attributes['aria-selected'], card.className.includes('is-selected'),
  ]), [
    ['Sphere', 'sphere', 'false', false],
    ['Cylinder', 'cylinder', 'true', true],
  ]);
  assert.equal(root.children[0].children[2].disabled, false);
});

test('updates selection position without changing card order', () => {
  const { root, controller } = panelFixture();
  const result = { status: 'ok', display_label: 'Square', candidates: [
    { id: 'cube', label: 'Cube' }, { id: 'prism', label: 'Square prism' },
  ] };
  controller.update(result);
  const track = root.children[1].children[0];
  controller.update({ ...result, selected_id: 'prism' });
  assert.equal(track.style.transform, 'translateX(-100%)');
  assert.deepEqual(track.children.map((card) => card.dataset.candidateId), ['cube', 'prism']);
  assert.deepEqual(track.children.map((card) => card.attributes['aria-selected']), ['false', 'true']);
});

test('search matches candidate names case-insensitively in Recommendation API order', () => {
  const searches = [];
  const { root, controller } = panelFixture({
    onSearch(query) { searches.push(query); },
  });
  // Same stable candidate IDs/order supplied by the shared circle recommendation.
  const recommendation = {
    status: 'ok', display_label: 'Circle', selected_id: 'solid.sphere',
    candidates: [
      { id: 'solid.sphere', label: 'Sphere' },
      { id: 'solid.ellipsoid', label: 'Ellipsoid' },
      { id: 'solid.cylinder', label: 'Cylinder' },
      { id: 'solid.cone', label: 'Cone' },
      { id: 'solid.torus', label: 'Torus' },
    ],
  };
  controller.update(recommendation);
  const search = root.children[0].children[3];
  search.value = 'CYL';
  search.listeners.input();

  const track = root.children[1].children[0];
  assert.deepEqual(track.children.map((card) => card.dataset.candidateId),
    ['solid.cylinder']);
  assert.deepEqual(searches, ['CYL']);
  assert.equal(root.children[1].children[1].hidden, true);
});

test('search with no results shows an empty state and clearing restores the API list', () => {
  const { root, controller } = panelFixture();
  const recommendation = {
    status: 'ok', display_label: 'Circle', selected_id: 'solid.sphere',
    candidates: [
      { id: 'solid.sphere', label: 'Sphere' },
      { id: 'solid.ellipsoid', label: 'Ellipsoid' },
      { id: 'solid.cylinder', label: 'Cylinder' },
    ],
  };
  controller.update(recommendation);
  const search = root.children[0].children[3];
  const track = root.children[1].children[0];

  search.value = 'pyramid';
  search.listeners.input();
  assert.deepEqual(track.children, []);
  assert.equal(root.children[1].children[1].textContent, 'No matching candidates');
  assert.equal(root.children[1].children[1].hidden, false);
  controller.updateHandInteraction({ x: 20, y: 20, closed_fist: true }, 10);

  search.value = '';
  search.listeners.input();
  assert.deepEqual(track.children.map((card) => card.dataset.candidateId),
    recommendation.candidates.map((candidate) => candidate.id));
  assert.equal(root.children[1].children[1].hidden, true);
});

test('clears/hides for uncertain, unsupported, empty, or malformed recommendations', () => {
  const { root, controller } = panelFixture();
  const valid = { status: 'ok', display_label: 'Square',
    candidates: [{ id: 'cube', label: 'Cube' }] };
  controller.update(valid);
  for (const result of [
    { status: 'uncertain', display_label: 'Uncertain', candidates: [] },
    { status: 'unsupported', display_label: 'Other', candidates: [] },
    { status: 'ok', display_label: 'Circle', candidates: [] },
    { status: 'ok', display_label: 'Circle', candidates: [{ id: '', label: 'bad' }] },
    null,
  ]) {
    controller.update(result);
    assert.equal(root.hidden, true);
    assert.deepEqual(root.children, []);
  }
});

test('fist dwell confirms only after one full second over the selected card', async () => {
  const confirmed = [];
  const { root, controller } = panelFixture({
    onConfirm(id) { confirmed.push(id); return true; },
  });
  controller.update({ status: 'ok', display_label: 'Circle', selected_id: 'sphere',
    candidates: [{ id: 'sphere', label: 'Sphere' }, { id: 'cone', label: 'Cone' }] });
  const selectedCard = root.children[1].children[0].children[0];
  selectedCard.rect = { left: 100, top: 100, right: 400, bottom: 400 };

  controller.updateHandInteraction({ x: 200, y: 200, closed_fist: true }, 10);
  controller.updateHandInteraction({ x: 200, y: 200, closed_fist: true }, 1009);
  assert.deepEqual(confirmed, []);
  assert.equal(root.hidden, false);
  assert.equal(selectedCard.children[2].style['--dwell-angle'], '359.64deg');

  controller.updateHandInteraction({ x: 200, y: 200, closed_fist: true }, 1010);
  controller.updateHandInteraction({ x: 200, y: 200, closed_fist: true }, 1100);
  assert.deepEqual(confirmed, ['sphere']);
  await Promise.resolve();
  assert.equal(root.hidden, true);
});

test('releasing the fist or moving away cancels and resets dwell progress', () => {
  const { root, controller } = panelFixture({ onConfirm: () => true });
  controller.update({ status: 'ok', display_label: 'Square', selected_id: 'cube',
    candidates: [{ id: 'cube', label: 'Cube' }] });
  const card = root.children[1].children[0].children[0];
  card.rect = { left: 50, top: 50, right: 300, bottom: 300 };

  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: true }, 0);
  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: true }, 400);
  assert.equal(card.children[2].style['--dwell-angle'], '144deg');
  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: false }, 450);
  assert.equal(card.children[2].style['--dwell-angle'], '0deg');

  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: true }, 500);
  controller.updateHandInteraction({ x: 400, y: 100, closed_fist: true }, 900);
  assert.equal(card.children[2].style['--dwell-angle'], '0deg');
  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: true }, 1000);
  controller.updateHandInteraction({ x: 100, y: 100, closed_fist: true }, 1999);
  assert.equal(root.hidden, false);
});

test('cancel control dismisses candidates through the application callback', () => {
  let cancellations = 0;
  const { root, controller } = panelFixture({
    onCancel() { cancellations += 1; },
  });
  controller.update({ status: 'ok', display_label: 'Triangle',
    candidates: [{ id: 'tetrahedron', label: 'Tetrahedron' }] });
  root.children[0].children[2].click();
  assert.equal(cancellations, 1);
});
