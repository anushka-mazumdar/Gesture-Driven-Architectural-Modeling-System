/** Horizontal, display-only carousel for shared recommendation results. */
export function createCandidatePanelController(root, { onConfirm, onCancel, onSearch } = {}) {
  let rosterKey = null;
  let cards = [];
  let allCandidates = [];
  let progressRings = [];
  let track = null;
  let position = null;
  let searchInput = null;
  let emptyState = null;
  let selectedCandidateId = null;
  let latestSelectedId = null;
  let dwellCandidateId = null;
  let dwellStartedAt = null;
  let confirming = false;
  const DWELL_MS = 1000;

  function resetDwell() {
    dwellCandidateId = null;
    dwellStartedAt = null;
    for (const ring of progressRings) {
      ring.style.setProperty('--dwell-angle', '0deg');
      ring.children[0].textContent = 'Hold fist';
    }
  }

  function clear() {
    rosterKey = null;
    cards = [];
    allCandidates = [];
    progressRings = [];
    track = null;
    position = null;
    searchInput = null;
    emptyState = null;
    selectedCandidateId = null;
    latestSelectedId = null;
    dwellCandidateId = null;
    dwellStartedAt = null;
    confirming = false;
    if (!root) return;
    root.hidden = true;
    root.replaceChildren();
  }

  function updateSelection(selectedId, count) {
    if (!count || !cards.length) {
      selectedCandidateId = null;
      resetDwell();
      track.style.transform = 'translateX(0)';
      position.textContent = `0 / ${count}`;
      return;
    }
    let selectedIndex = cards.findIndex((card) => card.dataset.candidateId === selectedId);
    if (selectedIndex < 0) selectedIndex = 0;
    const nextSelectedId = cards[selectedIndex].dataset.candidateId;
    if (nextSelectedId !== selectedCandidateId) resetDwell();
    selectedCandidateId = nextSelectedId;
    cards.forEach((card, index) => {
      const selected = index === selectedIndex;
      card.setAttribute('aria-selected', String(selected));
      card.className = `candidate-card${selected ? ' is-selected' : ''}`;
    });
    track.style.transform = `translateX(-${selectedIndex * 100}%)`;
    position.textContent = `${selectedIndex + 1} / ${count}`;
  }

  function renderFilteredCandidates(selectedId = latestSelectedId) {
    const query = searchInput.value.trim().toLocaleLowerCase();
    const visibleCandidates = allCandidates.filter((candidate) =>
      candidate.label.toLocaleLowerCase().includes(query)
    );
    progressRings = [];
    cards = visibleCandidates.map((candidate) => {
      const card = root.ownerDocument.createElement('article');
      card.className = 'candidate-card';
      card.dataset.candidateId = candidate.id;
      card.setAttribute('role', 'option');
      card.setAttribute('aria-selected', 'false');
      const label = root.ownerDocument.createElement('span');
      label.className = 'candidate-card__label';
      label.textContent = candidate.label;
      const id = root.ownerDocument.createElement('code');
      id.className = 'candidate-card__id';
      id.textContent = candidate.id;
      const ring = root.ownerDocument.createElement('div');
      ring.className = 'candidate-card__confirm-progress';
      ring.setAttribute('aria-hidden', 'true');
      const ringText = root.ownerDocument.createElement('span');
      ringText.textContent = 'Hold fist';
      ring.appendChild(ringText);
      progressRings.push(ring);
      card.append(label, id, ring);
      return card;
    });
    track.replaceChildren(...cards);
    emptyState.hidden = visibleCandidates.length > 0;
    emptyState.textContent = visibleCandidates.length ? '' : 'No matching candidates';
    updateSelection(selectedId, visibleCandidates.length);
  }

  function update(recommendation) {
    if (!root || !recommendation || recommendation.status !== 'ok'
        || typeof recommendation.display_label !== 'string'
        || !recommendation.display_label.trim()
        || !Array.isArray(recommendation.candidates)) {
      clear();
      return;
    }

    const candidates = recommendation.candidates;
    if (!candidates.length || candidates.some((candidate) =>
      !candidate || typeof candidate.id !== 'string' || !candidate.id.trim()
      || typeof candidate.label !== 'string' || !candidate.label.trim()
    )) {
      clear();
      return;
    }

    const nextRosterKey = JSON.stringify([
      recommendation.display_label,
      candidates.map(({ id, label }) => [id, label]),
    ]);
    if (nextRosterKey !== rosterKey) {
      clear();
      const doc = root.ownerDocument;
      const header = doc.createElement('header');
      header.className = 'candidate-panel__header';
      const heading = doc.createElement('div');
      heading.className = 'candidate-panel__heading';
      heading.textContent = 'Detected shape';
      const shape = doc.createElement('h2');
      shape.className = 'candidate-panel__shape';
      shape.textContent = recommendation.display_label;
      const cancel = doc.createElement('button');
      cancel.className = 'candidate-panel__cancel';
      cancel.type = 'button';
      cancel.title = 'Return to the 3D scene';
      cancel.setAttribute('aria-label', 'Cancel recommendation');
      cancel.textContent = '×';
      cancel.addEventListener('click', () => {
        if (onCancel) onCancel();
      });
      header.append(heading, shape, cancel);

      searchInput = doc.createElement('input');
      searchInput.className = 'candidate-panel__search';
      searchInput.type = 'search';
      searchInput.placeholder = 'Search 3D shapes';
      searchInput.setAttribute('aria-label', 'Search candidate names');
      searchInput.addEventListener('input', () => {
        renderFilteredCandidates();
        if (onSearch) onSearch(searchInput.value);
      });
      header.appendChild(searchInput);

      const viewport = doc.createElement('div');
      viewport.className = 'candidate-carousel__viewport';
      viewport.setAttribute('aria-label', '3D candidates');
      track = doc.createElement('div');
      track.className = 'candidate-carousel__track';
      track.setAttribute('role', 'listbox');
      track.setAttribute('aria-orientation', 'horizontal');
      emptyState = doc.createElement('div');
      emptyState.className = 'candidate-carousel__empty';
      emptyState.hidden = true;
      viewport.appendChild(track);
      viewport.appendChild(emptyState);

      const footer = doc.createElement('footer');
      footer.className = 'candidate-panel__footer';
      const hint = doc.createElement('span');
      hint.textContent = 'Swipe left or right to browse';
      position = doc.createElement('span');
      position.className = 'candidate-panel__position';
      footer.append(hint, position);

      root.replaceChildren(header, viewport, footer);
      root.hidden = false;
      rosterKey = nextRosterKey;
      allCandidates = candidates;
      searchInput.value = recommendation.search_query || '';
      latestSelectedId = recommendation.selected_id || candidates[0].id;
      renderFilteredCandidates(latestSelectedId);
    }

    root.hidden = false;
    const synchronizedQuery = recommendation.search_query || '';
    if (searchInput.value !== synchronizedQuery) {
      searchInput.value = synchronizedQuery;
      renderFilteredCandidates(recommendation.selected_id || latestSelectedId);
    }
    latestSelectedId = recommendation.selected_id || latestSelectedId || candidates[0].id;
    updateSelection(latestSelectedId, cards.length);
  }

  function updateHandInteraction(handState, now = performance.now()) {
    if (confirming) return;
    const { x, y, closed_fist } = handState || {};
    if (!root || root.hidden || !selectedCandidateId
        || !Number.isFinite(x) || !Number.isFinite(y) || !closed_fist) {
      resetDwell();
      return;
    }

    const selectedIndex = cards.findIndex(
      (card) => card.dataset.candidateId === selectedCandidateId
    );
    const rect = cards[selectedIndex]?.getBoundingClientRect();
    const overSelectedCard = rect
      && x >= rect.left && x <= rect.right
      && y >= rect.top && y <= rect.bottom;
    if (!overSelectedCard) {
      resetDwell();
      return;
    }

    if (dwellCandidateId !== selectedCandidateId || dwellStartedAt === null) {
      resetDwell();
      dwellCandidateId = selectedCandidateId;
      dwellStartedAt = now;
    }

    const elapsed = Math.max(0, now - dwellStartedAt);
    const progress = Math.min(elapsed / DWELL_MS, 1);
    const ring = progressRings[selectedIndex];
    ring.style.setProperty('--dwell-angle', `${progress * 360}deg`);
    ring.children[0].textContent = `${Math.ceil((DWELL_MS - elapsed) / 1000)}s`;

    if (elapsed < DWELL_MS || confirming) return;
    confirming = true;
    let confirmation;
    try {
      confirmation = onConfirm ? onConfirm(selectedCandidateId) : false;
    } catch (_error) {
      confirming = false;
      resetDwell();
      return;
    }
    Promise.resolve(confirmation).then((accepted) => {
      if (accepted) clear();
      else {
        confirming = false;
        resetDwell();
      }
    }).catch(() => {
      confirming = false;
      resetDwell();
    });
  }

  clear();
  return { update, updateHandInteraction, clear };
}
