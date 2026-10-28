# Gesture-Driven Architectural Modeling System

A desktop 3D modeling workspace controlled by hand gestures. Sketch a shape in front of a webcam, review shape-based 3D recommendations, and place, transform, and connect the resulting geometry in a live Three.js scene.

The application is built with Python for camera tracking, gesture interpretation, sketch processing, and geometry; a pywebview window hosts the workspace UI and Three.js renderer.

## Features

- **Webcam hand tracking:** MediaPipe Tasks tracks one hand and supplies landmarks to the gesture and rendering pipeline. The application mirrors a 640 × 480 camera image for interaction.
- **Freehand drawing:** Enable drawing with an open palm, then trace with the index finger. Stroke capture smooths and normalizes input before classifying it as an open path or closed shape.
- **Shape recognition:** A bundled Random Forest classifier recognizes circle, ellipse, triangle, square, rectangle, pentagon, and hexagon sketches. The shape taxonomy also defines larger open- and closed-stroke categories for recommendation and geometry workflows.
- **3D recommendations:** After a stroke is classified, the workspace presents a searchable, swipe-navigable carousel of candidate 3D forms. Hover the tracked hand over a candidate with a fist for one second to confirm it. Candidate geometry is built from the captured sketch where a constructor is implemented.
- **Object manipulation:** Use a peace sign over an object to select and manipulate it. Hand position moves it, wrist tilt rotates it, finger spread scales it, and vertical swipes adjust depth. A fast hand movement can detach an object from its snap assembly.
- **Anchor snapping and assemblies:** Optional snapping previews nearby compatible vertex, edge, and face anchors while manipulating. Releasing the manipulation commits the preview into a logical assembly; connected objects remain individually rendered and move together as an assembly.
- **Workspace controls:** Choose a stroke color, switch mesh/wireframe view, enable snapping, search candidates, and clear the scene. Touch input is shown as a future option and is not connected.
- **3D hand visualization:** A tracked hand mesh is rendered in the scene as a visual aid.

## Requirements

- Python with the packages listed in [`requirements.txt`](requirements.txt)
- A webcam
- Windows desktop is the primary target (camera capture requests DirectShow first)

The MediaPipe hand-landmarker model is included at `models/hand_landmarker.task`. If MediaPipe or the model is unavailable, the tracker falls back to a mock landmark source, but a webcam is still needed to provide frames.

## Run

From the project directory, create and activate a virtual environment, install the dependencies, and launch the app:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Allow camera access when prompted. The app opens one resizable workspace window containing the control panel and 3D viewport.

## Gesture controls

| Hand action | Result |
| --- | --- |
| Hold an open palm for about 1.5 seconds | Enable drawing |
| Index finger only | Draw a stroke while drawing is enabled |
| Stop drawing | Finish the stroke and open its recommendation carousel when classification succeeds |
| Swipe left or right while the carousel is open | Browse candidate forms |
| Hold a fist over the selected candidate for 1 second | Confirm the candidate |
| Peace sign over an existing object | Select and manipulate the object |
| Move the hand while manipulating | Move the object |
| Tilt the wrist / change finger spread | Rotate / scale the object |
| Swipe up or down while manipulating | Adjust object depth |
| Hold a fist over an object for 1 second | Delete the object |

The recommendation panel takes ownership of hand input while it is open, so drawing and scene manipulation do not run at the same time as candidate navigation or confirmation.

## Project layout

```text
.
├── main.py                    # Application startup and camera/gesture loop
├── webview_app/                # Single-window UI, styles, and Three.js client
├── vision/                     # MediaPipe tracking and landmark helpers
├── gestures/                   # Gesture recognition, stabilization, and timing
├── sketch/                     # Sketch-related UI and support code
├── shapes/                     # Stroke pipeline, taxonomy, recommendations, geometry
├── interaction/                # Selection, manipulation, placement, and snapping
├── render/                     # Three.js bridge, meshes, anchors, and hand rendering
├── models/                     # Hand landmark asset and trained shape classifiers
├── dataset/                     # Collected strokes, processed data, and ML splits
├── tools/                       # Dataset, training, and evaluation utilities
├── evaluation/                  # Saved classifier comparisons and metrics
└── tests/                       # Unit and integration coverage
```

## Shape-classifier evaluation

The bundled final Random Forest was trained on 282 samples and evaluated on a held-out test split of 61 samples. The saved evaluation reports **98.36% test accuracy** and **0.984 macro F1** across the seven supported classifier labels. These are results on this project’s recorded dataset; they do not guarantee the same accuracy for new users, cameras, or drawing conditions. Details are in [`evaluation/shape_random_forest_final_test_results.json`](evaluation/shape_random_forest_final_test_results.json) and model metadata in [`models/shape_random_forest_final_metadata.json`](models/shape_random_forest_final_metadata.json).

## Data and model workflow

The `tools/` directory contains scripts to collect strokes, preprocess and split data, build ML feature sets, train classifiers, and evaluate results. For example:

```powershell
python tools/build_ml_dataset.py
python tools/train_final_random_forest.py
python tools/evaluate_shape_classifier.py
```

Review each script's command-line options before running it. Training or dataset-building commands may regenerate files under `dataset/`, `models/`, or `evaluation/`.

## Current scope and limitations

- The UI is a desktop pywebview application; touch mode is not implemented.
- The shape classifier’s final evaluated model covers seven closed-shape labels. Open strokes and additional polygon types are handled by the broader stroke-processing and taxonomy code, but should not be read as extra labels in that reported classifier score.
- The taxonomy contains more candidate forms than the set of constructors necessarily supports. The geometry constructor code is the source of truth for which recommendations can currently produce a distinct mesh.
- Hand tracking and gesture reliability depend on camera quality, lighting, hand visibility, and the user’s drawing motion.
- Scene interaction is gesture-based; the viewport is not a general-purpose mouse-driven CAD editor.

## Development checks

The repository includes Python and JavaScript tests under `tests/` and smoke scripts under `scripts/`. Run an individual test with:

```powershell
python -m pytest tests/test_candidate_placement.py
```

Some test files exercise the JavaScript candidate panel with Node.js. The test suite is not run as part of the application startup.
