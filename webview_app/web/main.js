import * as THREE from './vendor/three.module.min.js';

// ---- Scene setup -----------------------------------------------------------

const container = document.getElementById('scene-container');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);

// Mesh vertices arrive in the same "pixel-space" units the previous
// PyOpenGL renderer used (camera_distance=500, near=1, far=3000, fovy=45,
// lookAt aimed at the origin) — match that here so real project meshes
// (which can span hundreds of units) land inside the view frustum.
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  1,
  3000
);
camera.position.set(0, 0, 500);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);
container.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(3, 5, 2);
scene.add(dirLight);

const grid = new THREE.GridHelper(800, 20, 0x334, 0x223);
grid.position.y = -100;
scene.add(grid);

// ---- Live scene: meshes pushed from Python --------------------------------
//
// render/threejs_renderer.py pushes a full snapshot of the current scene
// objects every frame. Each mesh is built into a THREE.BufferGeometry
// from vertices/normals/indices produced by render/primitives.py and gets
// the same position/rotation/scale TRS transform applied.

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

// Rebuilt in full on each push — object counts are small.
window.receiveMeshesFromPython = function (meshList) {
  for (const child of liveSceneGroup.children.slice()) {
    liveSceneGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }

  for (const data of meshList) {
    liveSceneGroup.add(buildMeshFromBridgeData(data));
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

// ---- UI --------------------------------------------------------------------

// Exit button → Python closes the pywebview window (which lets the app
// loop stop and release the webcam cleanly).
document.getElementById('exit-btn').addEventListener('click', () => {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.exit) {
    window.pywebview.api.exit();
  }
});

// ---- Render loop -----------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});