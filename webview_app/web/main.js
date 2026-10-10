import * as THREE from './vendor/three.module.min.js';
import { createCandidatePanelController } from './candidate_panel.mjs';

// ---- Scene setup -----------------------------------------------------------

const container = document.getElementById('scene-container');
let meshViewEnabled = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);

// Mesh vertices arrive in the same "pixel-space" units the previous
// PyOpenGL renderer used (camera_distance=500, near=1, far=3000, fovy=45,
// lookAt aimed at the origin) — match that here so real project meshes
// (which can span hundreds of units) land inside the view frustum.
const camera = new THREE.PerspectiveCamera(
  45,
  container.clientWidth / container.clientHeight,
  1,
  3000
);
camera.position.set(0, 0, 500);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);
container.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(3, 5, 2);
scene.add(dirLight);

// Procedural editor grid: the scene contains only a two-triangle plane; the
// repeating grid is drawn in its shader instead of allocating a huge line grid.
const grid = new THREE.Mesh(
  new THREE.PlaneGeometry(20000, 20000),
  new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    extensions: { derivatives: true },
    uniforms: {
      minorColor: { value: new THREE.Color(0x252d3b) },
      majorColor: { value: new THREE.Color(0x354153) },
    },
    vertexShader: `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 minorColor;
      uniform vec3 majorColor;
      varying vec3 vWorldPosition;
      float gridLine(vec2 coordinate, float spacing) {
        vec2 cell = coordinate / spacing;
        vec2 distanceToLine = abs(fract(cell - 0.5) - 0.5) / fwidth(cell);
        return 1.0 - min(min(distanceToLine.x, distanceToLine.y), 1.0);
      }
      void main() {
        vec2 plane = vWorldPosition.xz;
        float minor = gridLine(plane, 25.0);
        float major = gridLine(plane, 100.0);
        float fade = 1.0 - smoothstep(1300.0, 6500.0,
          distance(cameraPosition.xz, plane));
        float alpha = max(minor * 0.24, major * 0.52) * fade;
        vec3 color = mix(minorColor, majorColor, smoothstep(0.05, 0.4, major));
        gl_FragColor = vec4(color, alpha);
        if (gl_FragColor.a < 0.015) discard;
      }
    `,
  })
);
grid.rotation.x = -Math.PI / 2;
grid.position.y = -100;
grid.renderOrder = -2;
scene.add(grid);

// ---- Live scene: meshes pushed from Python --------------------------------
//
// render/threejs_renderer.py sends new mesh geometry once, then lightweight
// transform deltas. BufferGeometry is only built when a scene object is added.

function buildMeshFromBridgeData(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position', new THREE.BufferAttribute(new Float32Array(data.vertices), 3)
  );
  geometry.setAttribute(
    'normal', new THREE.BufferAttribute(new Float32Array(data.normals), 3)
  );
  geometry.setIndex(data.indices);

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(data.color[0], data.color[1], data.color[2]),
    metalness: 0.15,
    roughness: 0.6,
    // The old OpenGL path never enabled GL_CULL_FACE, so it always drew
    // both sides of every triangle regardless of winding order. Three.js
    // defaults to FrontSide (culls backfaces); a closed stroke can be
    // traced by the user in either winding direction, so keep both sides
    // visible instead of assuming one winding is always correct.
    side: THREE.DoubleSide,
    wireframe: meshViewEnabled,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(data.position[0], data.position[1], data.position[2]);
  // The renderer applies rotations in X, Y, Z order (degrees) — Three's
  // 'XYZ' Euler order matches that composition.
  mesh.rotation.set(
    THREE.MathUtils.degToRad(data.rotation[0]),
    THREE.MathUtils.degToRad(data.rotation[1]),
    THREE.MathUtils.degToRad(data.rotation[2]),
    'XYZ'
  );
  mesh.scale.set(data.scale[0], data.scale[1], data.scale[2]);
  return mesh;
}

const liveSceneGroup = new THREE.Group();
scene.add(liveSceneGroup);
const liveMeshesById = new Map();
const drawingPreviewGroup = new THREE.Group();
scene.add(drawingPreviewGroup);
const drawingLineGeometry = new THREE.BufferGeometry();
let drawingLineCapacity = 256;
const drawingStrokeHalfWidth = 1.35;
let drawingLinePositions = new Float32Array(drawingLineCapacity * 2 * 3);
let drawingLineIndices = new Uint32Array((drawingLineCapacity - 1) * 6);
for (let index = 0; index < drawingLineCapacity - 1; index += 1) {
  const first = index * 2;
  const next = first + 2;
  const offset = index * 6;
  drawingLineIndices.set([first, first + 1, next, first + 1, next + 1, next], offset);
}
let drawingLineAttribute = new THREE.BufferAttribute(drawingLinePositions, 3);
drawingLineAttribute.setUsage(THREE.DynamicDrawUsage);
drawingLineGeometry.setAttribute('position', drawingLineAttribute);
drawingLineGeometry.setIndex(new THREE.BufferAttribute(drawingLineIndices, 1));
drawingLineGeometry.setDrawRange(0, 0);
const drawingLine = new THREE.Mesh(
  drawingLineGeometry,
  new THREE.MeshBasicMaterial({
    color: 0x66bfff, transparent: true, opacity: 0.96,
    depthTest: false, depthWrite: false, side: THREE.DoubleSide,
  })
);
drawingLine.renderOrder = 20;
drawingPreviewGroup.add(drawingLine);
let lastPreviewPointCount = -1;
let lastPreviewLastPoint = null;
let lastPreviewColor = '';
const cursorMaterial = new THREE.MeshBasicMaterial({
  color: 0xf3f7ff, depthTest: false, depthWrite: false,
});
const cursorRing = new THREE.Mesh(new THREE.RingGeometry(7.5, 9.5, 32), cursorMaterial);
const cursorDot = new THREE.Mesh(new THREE.CircleGeometry(2, 16), cursorMaterial);
let deleteProgressArc = null;
cursorRing.renderOrder = 21;
cursorDot.renderOrder = 21;
drawingPreviewGroup.add(cursorRing, cursorDot);
const snapPreviewGroup = new THREE.Group();
scene.add(snapPreviewGroup);
const snapPreviewHud = document.getElementById('snap-preview-hud');

function clearSnapPreview() {
  for (const child of snapPreviewGroup.children.slice()) {
    snapPreviewGroup.remove(child);
    child.geometry?.dispose();
    if (Array.isArray(child.material)) child.material.forEach((item) => item.dispose());
    else child.material?.dispose();
  }
}

// This layer is deliberately independent of liveSceneGroup: it visualizes a
// possible connection but never changes either object's transform.
window.receiveSnapPreviewFromPython = function (preview) {
  clearSnapPreview();
  if (!preview || !Array.isArray(preview.moving_position)
      || !Array.isArray(preview.target_position)) {
    if (snapPreviewHud) snapPreviewHud.hidden = true;
    return;
  }

  const moving = new THREE.Vector3(...preview.moving_position);
  const target = new THREE.Vector3(...preview.target_position);
  for (const [point, radius] of [[moving, 5], [target, 8]]) {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 16, 12),
      new THREE.MeshBasicMaterial({
        color: 0x64dcff, depthTest: false, depthWrite: false,
        transparent: true, opacity: 0.92,
      })
    );
    marker.position.copy(point);
    marker.renderOrder = 30;
    snapPreviewGroup.add(marker);
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.55, 16, 12),
      new THREE.MeshBasicMaterial({
        color: 0x64dcff, wireframe: true, transparent: true,
        opacity: 0.58, depthTest: false, depthWrite: false,
      })
    );
    halo.position.copy(point);
    halo.renderOrder = 29;
    snapPreviewGroup.add(halo);
  }
  const connector = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([moving, target]),
    new THREE.LineDashedMaterial({
      color: 0x8be7ff, dashSize: 5, gapSize: 3,
      transparent: true, opacity: 0.95, depthTest: false,
    })
  );
  connector.computeLineDistances();
  connector.renderOrder = 31;
  snapPreviewGroup.add(connector);

  if (snapPreviewHud) {
    const kind = typeof preview.kind === 'string' ? preview.kind : 'anchor';
    snapPreviewHud.textContent = `SNAP PREVIEW · ${kind.toUpperCase()}`;
    snapPreviewHud.hidden = false;
  }
};

window.receiveMeshesFromPython = function (delta) {
  if (!delta || Array.isArray(delta)) return;
  for (const id of delta.removed || []) {
    const mesh = liveMeshesById.get(id);
    if (!mesh) continue;
    liveSceneGroup.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    liveMeshesById.delete(id);
  }
  for (const data of delta.added || []) {
    const previous = liveMeshesById.get(data.id);
    if (previous) {
      liveSceneGroup.remove(previous);
      previous.geometry.dispose();
      previous.material.dispose();
    }
    const mesh = buildMeshFromBridgeData(data);
    liveMeshesById.set(data.id, mesh);
    liveSceneGroup.add(mesh);
  }
  for (const transform of delta.transforms || []) {
    const mesh = liveMeshesById.get(transform.id);
    if (!mesh) continue;
    mesh.position.set(...transform.position);
    mesh.rotation.set(
      THREE.MathUtils.degToRad(transform.rotation[0]),
      THREE.MathUtils.degToRad(transform.rotation[1]),
      THREE.MathUtils.degToRad(transform.rotation[2]), 'XYZ'
    );
    mesh.scale.set(...transform.scale);
  }
};

// ---- Shape classification HUD ----------------------------------------------

const shapeHud = document.getElementById('shape-hud');

window.receiveHudFromPython = function (text) {
  if (!shapeHud) return;
  if (text) {
    shapeHud.textContent = text;
    shapeHud.style.display = 'block';
  } else {
    shapeHud.style.display = 'none';
  }
};

const drawHint = document.getElementById('draw-hint');

window.receiveDrawingStateFromPython = function (state) {
  if (!state) return;
  const points = Array.isArray(state.points) ? state.points : [];
  const lastPoint = points.length ? points[points.length - 1] : null;
  const colorKey = Array.isArray(state.stroke_color) ? state.stroke_color.join(',') : '';
  const pointsChanged = points.length !== lastPreviewPointCount
    || (lastPoint && (!lastPreviewLastPoint
      || lastPoint[0] !== lastPreviewLastPoint[0]
      || lastPoint[1] !== lastPreviewLastPoint[1]));
  if (pointsChanged) {
    if (points.length > drawingLineCapacity) {
      while (drawingLineCapacity < points.length) drawingLineCapacity *= 2;
      drawingLinePositions = new Float32Array(drawingLineCapacity * 2 * 3);
      drawingLineIndices = new Uint32Array((drawingLineCapacity - 1) * 6);
      for (let index = 0; index < drawingLineCapacity - 1; index += 1) {
        const first = index * 2;
        const next = first + 2;
        const offset = index * 6;
        drawingLineIndices.set([first, first + 1, next, first + 1, next + 1, next], offset);
      }
      drawingLineAttribute = new THREE.BufferAttribute(drawingLinePositions, 3);
      drawingLineAttribute.setUsage(THREE.DynamicDrawUsage);
      drawingLineGeometry.setAttribute('position', drawingLineAttribute);
      drawingLineGeometry.setIndex(new THREE.BufferAttribute(drawingLineIndices, 1));
    }
    for (let index = 0; index < points.length; index += 1) {
      const previous = points[Math.max(0, index - 1)];
      const next = points[Math.min(points.length - 1, index + 1)];
      const tangentX = next[0] - previous[0];
      const tangentY = next[1] - previous[1];
      const tangentLength = Math.hypot(tangentX, tangentY) || 1;
      const normalX = -tangentY / tangentLength * drawingStrokeHalfWidth;
      const normalY = tangentX / tangentLength * drawingStrokeHalfWidth;
      const centerX = points[index][0] - 320;
      const centerY = 240 - points[index][1];
      const offset = index * 6;
      drawingLinePositions[offset] = centerX + normalX;
      drawingLinePositions[offset + 1] = centerY - normalY;
      drawingLinePositions[offset + 2] = 8;
      drawingLinePositions[offset + 3] = centerX - normalX;
      drawingLinePositions[offset + 4] = centerY + normalY;
      drawingLinePositions[offset + 5] = 8;
    }
    drawingLineAttribute.needsUpdate = true;
    drawingLineGeometry.setDrawRange(0, Math.max(0, points.length - 1) * 6);
    lastPreviewPointCount = points.length;
    lastPreviewLastPoint = lastPoint ? [...lastPoint] : null;
  }
  if (colorKey !== lastPreviewColor) {
    drawingLine.material.color.set(Array.isArray(state.stroke_color)
      && state.stroke_color.length === 3
      ? new THREE.Color(...state.stroke_color) : 0x66bfff);
    lastPreviewColor = colorKey;
  }

  if (Array.isArray(state.cursor) && state.cursor.length === 2) {
    const x = state.cursor[0] * 640 - 320;
    const y = 240 - state.cursor[1] * 480;
    const cursorColor = state.near_object ? 0x64ffc3 : 0xf3f7ff;
    cursorMaterial.color.set(cursorColor);
    cursorRing.position.set(x, y, 12);
    cursorDot.position.set(x, y, 12);

    if (state.delete_progress > 0) {
      const arcPoints = [];
      const segments = Math.max(3, Math.ceil(48 * state.delete_progress));
      for (let index = 0; index <= segments; index += 1) {
        const angle = -Math.PI / 2 + Math.PI * 2 * state.delete_progress * index / segments;
        arcPoints.push(new THREE.Vector3(x + Math.cos(angle) * 15,
          y + Math.sin(angle) * 15, 13));
      }
      if (deleteProgressArc) {
        drawingPreviewGroup.remove(deleteProgressArc);
        deleteProgressArc.geometry.dispose();
        deleteProgressArc.material.dispose();
      }
      const arc = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(arcPoints),
        new THREE.LineBasicMaterial({ color: 0xff5a5a, depthTest: false })
      );
      arc.renderOrder = 22;
      drawingPreviewGroup.add(arc);
      deleteProgressArc = arc;
    }
    cursorRing.visible = true;
    cursorDot.visible = true;
  } else {
    cursorRing.visible = false;
    cursorDot.visible = false;
  }
  if (!(state.delete_progress > 0) && deleteProgressArc) {
    drawingPreviewGroup.remove(deleteProgressArc);
    deleteProgressArc.geometry.dispose();
    deleteProgressArc.material.dispose();
    deleteProgressArc = null;
  }

  drawHint.hidden = state.hint_visible === false;
  drawHint.textContent = state.drawing_enabled
    ? 'Index finger to draw · Open palm to finish'
    : 'Open palm to enable sketching · Index finger to draw';
};

// Display-only candidate list supplied by the shared Recommendation API.
const candidatePanel = createCandidatePanelController(
  document.getElementById('candidate-panel'),
  {
    onConfirm(candidateId) {
      const confirmCandidate = window.pywebview?.api?.confirm_candidate;
      return confirmCandidate ? confirmCandidate(candidateId) : false;
    },
    onCancel() {
      const cancelCandidates = window.pywebview?.api?.cancel_candidate_panel;
      return cancelCandidates ? cancelCandidates() : false;
    },
    onSearch(query) {
      const searchCandidates = window.pywebview?.api?.search_candidates;
      return searchCandidates ? searchCandidates(query) : false;
    },
  }
);
window.receiveCandidatePanelFromPython = function (recommendation) {
  document.getElementById('workspace-panel')?.classList.toggle(
    'is-recommendation-active', Boolean(recommendation && recommendation.status === 'ok')
  );
  candidatePanel.update(recommendation);
};
window.receiveCandidateHandStateFromPython = function (handState) {
  if (!handState) {
    candidatePanel.updateHandInteraction(null);
    return;
  }
  const sceneBounds = container.getBoundingClientRect();
  candidatePanel.updateHandInteraction({
    x: sceneBounds.left + handState.x * sceneBounds.width,
    y: sceneBounds.top + handState.y * sceneBounds.height,
    closed_fist: handState.closed_fist,
  });
};

// Workspace settings are mirrored from Python; the UI never edits gesture or
// snap state locally. Recommendation mode temporarily locks normal controls.
window.receiveWorkspaceSettingsFromPython = function (settings) {
  if (!settings) return;
  const snapToggle = document.getElementById('snapping-toggle');
  const meshToggle = document.getElementById('mesh-view-toggle');
  if (snapToggle) snapToggle.checked = Boolean(settings.snapping_enabled);
  if (Array.isArray(settings.stroke_color) && settings.stroke_color.length === 3) {
    const hex = `#${settings.stroke_color.map((component) =>
      Math.round(Math.max(0, Math.min(1, component)) * 255)
        .toString(16).padStart(2, '0')
    ).join('')}`;
    const selected = [...document.querySelectorAll('.color-swatch')]
      .find((swatch) => swatch.dataset.color.toLowerCase() === hex);
    if (selected) {
      document.querySelector('.color-swatch.is-selected')?.classList.remove('is-selected');
      selected.classList.add('is-selected');
      workspacePanel?.style.setProperty('--workspace-accent', hex);
      const colorName = selected.getAttribute('aria-label') || 'CUSTOM';
      document.getElementById('color-value').textContent = colorName.toUpperCase();
    }
  }
  const nextMeshView = Boolean(settings.mesh_view);
  if (meshToggle) meshToggle.checked = nextMeshView;
  if (nextMeshView !== meshViewEnabled) {
    meshViewEnabled = nextMeshView;
    liveSceneGroup.traverse((object) => {
      if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => { material.wireframe = meshViewEnabled; });
      }
    });
  }
};

const workspacePanel = document.getElementById('workspace-panel');
const snappingToggle = document.getElementById('snapping-toggle');
const meshViewToggle = document.getElementById('mesh-view-toggle');
const drawModeButton = document.getElementById('draw-mode-button');
const clearSceneButton = document.getElementById('clear-scene-button');

function reportSettingFailure(toggle, previousValue) {
  toggle.checked = previousValue;
  toggle.title = 'Could not update this setting';
}

snappingToggle?.addEventListener('change', async () => {
  const previousValue = !snappingToggle.checked;
  try {
    const setEnabled = window.pywebview?.api?.set_snapping_enabled;
    if (!setEnabled) throw new Error('Python bridge unavailable');
    const enabled = await setEnabled(snappingToggle.checked);
    snappingToggle.checked = Boolean(enabled);
    snappingToggle.title = enabled ? 'Snapping enabled' : 'Snapping disabled';
  } catch (_error) {
    reportSettingFailure(snappingToggle, previousValue);
  }
});

meshViewToggle?.addEventListener('change', async () => {
  const previousValue = !meshViewToggle.checked;
  try {
    const setMeshView = window.pywebview?.api?.set_mesh_view;
    if (!setMeshView) throw new Error('Python bridge unavailable');
    await setMeshView(meshViewToggle.checked);
  } catch (_error) {
    reportSettingFailure(meshViewToggle, previousValue);
  }
});

drawModeButton?.addEventListener('click', async () => {
  try {
    const setDrawing = window.pywebview?.api?.set_drawing_enabled;
    if (!setDrawing) return;
    const enabled = await setDrawing(true);
    if (enabled) {
      drawModeButton.classList.add('is-active');
      drawModeButton.setAttribute('aria-pressed', 'true');
    }
  } catch (_error) {
    // Gesture-based drawing remains available if the optional UI bridge fails.
  }
});

clearSceneButton?.addEventListener('click', async () => {
  clearSceneButton.disabled = true;
  try {
    const clearScene = window.pywebview?.api?.clear_scene;
    if (!clearScene) throw new Error('Python bridge unavailable');
    const cleared = await clearScene();
    if (!cleared) throw new Error('Scene could not be cleared');
    clearSceneButton.title = 'Scene cleared';
  } catch (_error) {
    clearSceneButton.title = 'Could not clear the scene';
  } finally {
    clearSceneButton.disabled = false;
  }
});

document.querySelectorAll('.color-swatch').forEach((swatch) => {
  swatch.addEventListener('click', async () => {
    const previous = document.querySelector('.color-swatch.is-selected');
    try {
      const setColor = window.pywebview?.api?.set_stroke_color;
      if (!setColor) throw new Error('Python bridge unavailable');
      const accepted = await setColor(swatch.dataset.color);
      if (!Array.isArray(accepted)) throw new Error('Color was rejected');
      previous?.classList.remove('is-selected');
      swatch.classList.add('is-selected');
      workspacePanel?.style.setProperty('--workspace-accent', swatch.dataset.color);
      const colorName = swatch.getAttribute('aria-label') || 'CUSTOM';
      document.getElementById('color-value').textContent = colorName.toUpperCase();
    } catch (_error) {
      // Keep Python's active shape color authoritative if the bridge fails.
    }
  });
});

const workspaceSearch = document.getElementById('workspace-search-input');
window.addEventListener('keydown', (event) => {
  if (event.key === '/' && !event.target.matches('input, textarea')) {
    event.preventDefault();
    workspaceSearch?.focus();
  }
  if (event.key === 'Escape' && document.activeElement === workspaceSearch) {
    workspaceSearch.blur();
  }
});

// ---- Tracked-hand overlay --------------------------------------------------
//
// A smooth, matte-gray "glove" style hand instead of separate segmented
// tubes: each finger is one continuous rounded tube through its joints
// (THREE.CatmullRomCurve3 + TubeGeometry, so there are no visible seams
// at the knuckles), plus a soft ellipsoid palm and a capped wrist stub
// that both orient/scale to the tracked hand every frame. Python
// (render/hand_mesh.py) remains authoritative for the tracked landmark
// positions — only those 21 points cross the bridge each frame
// (render/mesh_bridge.py:serialize_hand); the finger/palm layout below
// is a fixed constant, not per-frame state.

const HAND_COLOR = new THREE.Color(0xd6d6d6);

function handMaterial() {
  return new THREE.MeshStandardMaterial({
    color: HAND_COLOR,
    roughness: 0.6,
    metalness: 0.05,
  });
}

// [landmark indices from base to tip, tube radius, tip cap radius]
const FINGER_CHAINS = [
  { points: [1, 2, 3, 4],       radius: 6.5, tipRadius: 4.0 }, // thumb
  { points: [5, 6, 7, 8],       radius: 6.0, tipRadius: 4.5 }, // index
  { points: [9, 10, 11, 12],    radius: 6.3, tipRadius: 4.6 }, // middle
  { points: [13, 14, 15, 16],   radius: 5.8, tipRadius: 4.2 }, // ring
  { points: [17, 18, 19, 20],   radius: 5.2, tipRadius: 3.8 }, // pinky
];

const handGroup = new THREE.Group();
scene.add(handGroup);

function clearHandGroup() {
  for (const child of handGroup.children.slice()) {
    handGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }
}

function addFingerTube(points, radius, tipRadius) {
  const pts = points.map((p) => new THREE.Vector3(...p));
  const curve = new THREE.CatmullRomCurve3(pts);
  const geometry = new THREE.TubeGeometry(curve, 16, radius, 10, false);
  handGroup.add(new THREE.Mesh(geometry, handMaterial()));

  // round caps at both ends so the tube blends into the tip and palm
  const baseCap = new THREE.Mesh(new THREE.SphereGeometry(radius, 10, 8), handMaterial());
  baseCap.position.copy(pts[0]);
  handGroup.add(baseCap);

  const tipCap = new THREE.Mesh(new THREE.SphereGeometry(tipRadius, 10, 8), handMaterial());
  tipCap.position.copy(pts[pts.length - 1]);
  handGroup.add(tipCap);
}

function addPalm(landmarks) {
  const wrist    = new THREE.Vector3(...landmarks[0]);
  const knuckles = [5, 9, 13, 17].map((i) => new THREE.Vector3(...landmarks[i]));
  const knuckleCenter = knuckles
    .reduce((a, b) => a.add(b), new THREE.Vector3())
    .multiplyScalar(1 / knuckles.length);
  const palmCenter = wrist.clone().add(knuckleCenter).multiplyScalar(0.5);

  const lengthAxis = new THREE.Vector3().subVectors(knuckleCenter, wrist);
  const length = lengthAxis.length() || 1;
  lengthAxis.normalize();

  const widthAxis = new THREE.Vector3().subVectors(knuckles[3], knuckles[0]); // pinky - index knuckle
  const width = widthAxis.length() || 1;
  widthAxis.normalize();

  const normalAxis = new THREE.Vector3().crossVectors(widthAxis, lengthAxis).normalize();
  widthAxis.crossVectors(lengthAxis, normalAxis).normalize(); // re-orthogonalize

  const basis = new THREE.Matrix4().makeBasis(widthAxis, lengthAxis, normalAxis);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 16), handMaterial());
  mesh.position.copy(palmCenter);
  mesh.quaternion.setFromRotationMatrix(basis);
  mesh.scale.set(width * 0.62, length * 0.62, length * 0.32);
  handGroup.add(mesh);

  return { wrist, lengthAxis, length, width };
}

function addWristStub(palm) {
  const back   = palm.wrist.clone().addScaledVector(palm.lengthAxis, -palm.length * 0.55);
  const radius = palm.width * 0.28;
  const geometry = new THREE.CylinderGeometry(radius, radius * 1.05, palm.length * 0.55, 14);
  const mesh = new THREE.Mesh(geometry, handMaterial());
  mesh.position.copy(palm.wrist).add(back).multiplyScalar(0.5);
  const dir = new THREE.Vector3().subVectors(palm.wrist, back).normalize();
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  handGroup.add(mesh);
}

window.receiveHandFromPython = function (landmarks) {
  clearHandGroup();
  if (!landmarks || landmarks.length !== 21) return;

  const palm = addPalm(landmarks);
  addWristStub(palm);

  for (const { points, radius, tipRadius } of FINGER_CHAINS) {
    addFingerTube(points.map((i) => landmarks[i]), radius, tipRadius);
  }
};

// Python batches current state into one call so its tracking loop never waits
// on several separate WebView round trips.
window.receiveFrameFromPython = function (frame) {
  if (!frame) return;
  window.receiveMeshesFromPython?.(frame.meshes);
  window.receiveSnapPreviewFromPython?.(frame.snap_preview);
  window.receiveWorkspaceSettingsFromPython?.(frame.workspace_settings);
  window.receiveHandFromPython?.(frame.hand);
  window.receiveHudFromPython?.(frame.hud);
  window.receiveCandidatePanelFromPython?.(frame.candidate_panel);
  window.receiveCandidateHandStateFromPython?.(frame.candidate_hand_state);
  window.receiveDrawingStateFromPython?.(frame.drawing);
};

// ---- UI --------------------------------------------------------------------

// Exit button → Python closes the pywebview window (which lets the app
// loop stop and release the webcam cleanly).
document.getElementById('exit-btn').addEventListener('click', () => {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.exit) {
    window.pywebview.api.exit();
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    window.pywebview?.api?.exit?.();
  }
});

// ---- Render loop -----------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});
