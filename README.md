# Gesture-Driven 2D-to-3D Architectural Modeling System

## 📌 Project Overview
This project presents a **CPU-optimized, monocular vision-based gesture interface** that allows users to sketch 2D shapes in mid-air using a webcam and convert them into manipulable 3D architectural primitives. The system emphasizes **intentional interaction**, **predictable behavior**, and **academic rigor**, making it suitable as a **final-year engineering project** and a strong **GitHub portfolio project**.

Inspired by futuristic gesture interfaces (e.g., Iron Man-style UI metaphors), this system is **not a hologram**. Instead, it is a **Human–Computer Interaction (HCI) and Computer Vision prototype** designed for reliability on **non-GPU hardware**.

---

## 🎯 Key Objectives
- Enable mid-air **2D sketching** using hand gestures
- Convert 2D sketches into valid **3D shapes**
- Allow **gesture-based manipulation** (move, rotate, scale)
- Support **intentional stacking and detachment** of objects
- Run entirely on **CPU-only systems**
- Be **defensible in viva** and suitable for a **research report**

---

## 🧠 Core Features
- Single-hand gesture interaction (CPU-optimized)
- Camera-aligned virtual sketch plane
- Rule-based gesture recognition (no ML inference during runtime)
- 2D-to-3D shape mapping with user confirmation
- Face-based snapping and reversible stacking
- Deterministic, explainable system behavior

---

## 🖐️ Gesture Summary

| Feature | Gesture |
|------|--------|
| Enter Sketch Mode | Open palm (3s) |
| Draw Shape | Pinch + move |
| Finish Drawing | Release pinch |
| Select Object | Pinch + hold (1s) |
| Move Object | One-finger swipe |
| Move Depth | Two-finger vertical swipe |
| Rotate | Wrist rotation |
| Scale | Pinch open / close |
| Stack | Bring object near another |
| Detach | Pinch + pull away |
| Delete | Closed fist (2s) |

---

## 🏗️ System Architecture

```
Webcam Feed
   ↓
Hand Detection (MediaPipe)
   ↓
Gesture State Machine
   ↓
Sketch / Manipulation Logic
   ↓
3D Rendering (OpenGL)
```

---

## 🧩 Technology Stack

### Software
- Python 3.9+
- OpenCV
- MediaPipe Hands
- NumPy
- PyOpenGL

### Hardware
- Standard webcam
- CPU-only laptop/desktop
- No GPU required

---

## 📂 Project Structure

```
gesture_architecture/
│
├── vision/          # Hand tracking & landmarks
├── sketch/          # 2D sketch panel & contour logic
├── gestures/        # Gesture rules & state machine
├── shapes/          # 2D recognition & 3D generation
├── interaction/     # Selection, snapping, manipulation
├── render/          # OpenGL renderer & primitives
├── main.py
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/yourusername/gesture-architecture.git
cd gesture-architecture
```

### 2. Create virtual environment
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the application
```
python main.py
```

---

## 🧪 Performance Constraints
- Webcam resolution: **640×480**
- Vision processing: ~15 FPS
- Rendering: ~30 FPS
- Wireframe / flat-shaded geometry only

---

## 📊 Evaluation Metrics (for Report)
- Gesture recognition accuracy
- Shape classification success rate
- Interaction latency
- Frames per second (FPS)
- User task completion time

---

## 🚧 Limitations
- No real depth perception (monocular only)
- No physics simulation
- No GPU-based visual effects
- Limited gesture vocabulary by design

---

## 🔮 Future Enhancements
- Optional ML-based gesture classification
- Kalman filtering for smoother tracking
- Multi-user study
- VR/AR headset integration
- Depth camera support

---

## 🎓 Academic Positioning
This project is best framed as:
- **Gesture-Based Human–Computer Interaction System**
- Supported by **Computer Vision**
- Applied to **Architectural Sketching and Spatial Modeling**

---

## 📜 License
This project is intended for **academic and educational use**.

---

## 🙌 Acknowledgements
- MediaPipe by Google
- OpenCV community
- OpenGL documentation

---

## ✅ Final Note
This project prioritizes **clarity, control, and correctness** over flashy visuals. It is intentionally scoped to be **implementable, explainable, and defensible** in a final-year academic setting.

