# ✋ Gesture-Driven 3D Architectural Modeling System

A real-time, gesture-controlled 3D modeling environment that enables users to sketch, generate, and manipulate 3D objects using only hand movements captured via a webcam.

---

## 🚀 Overview

This project replaces traditional input devices (mouse/keyboard) with **intuitive hand gestures**, allowing users to:

* Draw 2D strokes in air
* Convert sketches into 3D geometry
* Manipulate objects in real time
* Visualize hand motion using a holographic mesh overlay

Built using **Computer Vision + OpenGL**, the system focuses on responsiveness, stability, and clean interaction design.

---

## 🧠 Core Features

### ✍️ Gesture-Based Drawing

* Draw using **index finger**
* Smooth stroke interpolation with jitter filtering
* Automatic stroke completion and cleanup

---

### 🧊 3D Shape Generation

* Open strokes → **Ribbon (tube) meshes**
* Closed strokes → **Extruded polygon meshes**
* Dynamic scaling based on stroke dimensions

---

### 🎮 Real-Time Object Manipulation

Controlled via **peace sign gesture**:

* Move → Hand position
* Rotate → Wrist tilt
* Scale → Finger spread
* Depth → Swipe gestures

---

### 🧲 Snap-Ready Architecture

* Supports **group-based snapping system**
* Transformation propagation across grouped objects
* Currently disabled for clean independent interaction

---

### ✋ Holographic Hand Mesh Overlay

* Real-time **wireframe hand visualization**
* Based on 21 landmark points
* Styled as a **sci-fi holographic mesh**
* Positioned as a non-intrusive bottom-right overlay
* Completely independent from gesture detection logic

---

### 🧠 Gesture Intelligence Layer

* Gesture stabilization (buffer-based)
* Cooldown system to prevent flickering
* Hold detection for intentional actions (e.g., delete)

---

### ❌ Delete Mechanism

* Hold **fist over object** for 1 second
* Visual progress feedback
* Safe exclusion of hand mesh from deletion

---

## 🏗️ Project Structure

```
Gesture-Driven-Architectural-Modeling-System/
│
├── gestures/        # Gesture detection, stabilization, state logic
├── interaction/     # Object selection, manipulation, snapping
├── render/          # OpenGL renderer + hand mesh
├── shapes/          # Stroke processing + 3D mesh generation
├── sketch/          # Drawing panel UI
├── vision/          # Hand tracking + landmark utilities
│
└── main.py          # Main application loop
```

---

## ⚙️ Tech Stack

* **Python**
* **OpenCV** (camera + UI)
* **MediaPipe** (hand tracking)
* **OpenGL (PyOpenGL)** (3D rendering)
* **NumPy** (math + geometry)

---

## 🎯 Gesture Mapping

| Gesture     | Action                |
| ----------- | --------------------- |
| Index Only  | Draw                  |
| Peace Sign  | Manipulate Object     |
| Open Palm   | Activate Panel / Exit |
| Fist (Hold) | Delete Object         |
| Swipe       | Depth Control         |

---

## 🧪 Key Design Decisions

* ❌ Removed complex mesh blending → improved stability
* ❌ Disabled auto-snapping → better user control
* ✅ Prioritized **clean architecture over visual gimmicks**
* ✅ Simplified hand mesh for real-time performance

---

## ⚠️ Known Limitations

* No physics-based interaction
* Approximate depth perception
* Gesture ambiguity in extreme lighting
* Hand mesh is wireframe (not full surface mesh)

---

## 🔮 Future Improvements

* Shader-based glow / bloom effects
* True surface mesh hand (triangulated)
* Multi-hand interaction
* ML-based gesture classification
* Physics-based snapping

---

## 🏁 Status

✅ Core system stable
✅ Gesture pipeline optimized
✅ Hand mesh rendering finalized

🚧 Future work focuses on realism and UX enhancements

---

## 📸 Demo (optional)

*Add screenshots / demo GIFs here*

---

## 📌 Author Notes

This project evolved through multiple iterations focusing on:

* Stability over complexity
* Real-time responsiveness
* Clean interaction design

The final system reflects a **minimal, performant, and extensible architecture** for gesture-driven 3D interaction.

---

## 📜 License

MIT License (or your preferred license)

---
