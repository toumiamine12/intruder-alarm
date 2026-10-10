# Desk Intruder Alarm

![demo](docs/demo.gif)

A webcam watches your desk. When something moves inside the zone you draw, a red **INTRUDER** banner appears,
the moving object gets boxed, a bell rings and a timestamped snapshot is saved.

The core image-processing pipeline (background model, histogram, Otsu threshold, temporal filter) is
**written by hand in C++**, and a unit test checks it against OpenCV's own implementation.

Built as a portfolio project for my **Image Processing (Bildverarbeitung)** module.

## Features

- Live motion detection at webcam frame rate on a normal laptop CPU
- Detection zone: drag a rectangle with the mouse, so only motion inside it counts
- Alarm: red banner, red box around the intruder, a bell on trigger, a snapshot every 3 s to `snapshots/`
- Debug view (`d`): difference image, binary mask, and the histogram with the chosen threshold
- No false alarms from sensor noise: a minimum threshold, morphology and a 5-frame confirmation

## Pipeline

```
webcam frame (BGR)
  → grayscale + Gaussian blur (5×5)
  → diff = |frame − background|              ← BackgroundModel::diff
  → T = Otsu threshold of diff (min 25)      ← otsuThreshold
  → binary mask = diff > T                   ← thresholdMask
  → morphological opening 3×3, closing 7×7
  → keep only the detection zone
  → contours with area ≥ 1500 px → bounding box
  → motion 5 frames in a row? → ALARM        ← MotionDebouncer
  → update background, except under the box  ← BackgroundModel::update + makeSkipMask
```

## Image-processing topics used

| Topic | Where |
|---|---|
| Grayscale conversion & Gaussian smoothing | `main.cpp` |
| Histogram | `histogram()` in [`src/motion.cpp`](src/motion.cpp) |
| Automatic thresholding (Otsu's method) | `otsuThreshold()` |
| Binary segmentation | `thresholdMask()` |
| Background modelling (running average) | `BackgroundModel` |
| Morphology (opening / closing) | `main.cpp` |
| Contours & region features (area, bounding box) | `main.cpp` |
| Temporal filtering (hysteresis over frames) | `MotionDebouncer` |

## Written by hand vs. OpenCV

| By hand ([`src/motion.cpp`](src/motion.cpp)) | OpenCV |
|---|---|
| Running-average background model with a skip mask | Camera capture, color → gray |
| Absolute difference image | Gaussian blur |
| 256-bin histogram + Otsu threshold | Morphological open / close |
| Thresholding to a binary mask | `findContours`, `boundingRect` |
| Motion debouncer (temporal filter) | Drawing, windows, mouse, `imwrite` |

`tests/test_motion.cpp` checks each piece on synthetic images. One check compares my Otsu with
`cv::threshold(..., THRESH_OTSU)` and requires the result to be within ±1 gray level.

### Two details worth mentioning

- **Standing still doesn't hide you.** While the alarm is on, the background stops learning under the
  intruder's box, so a person who freezes doesn't slowly fade into the background.
- **Light switches don't jam it.** If the "motion" covers more than half the frame (for example, a lamp
  turned on), the whole background keeps learning, so the alarm clears by itself.

## Build & run (Fedora)

```bash
sudo dnf install -y opencv-devel cmake gcc-c++
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure   # unit tests
./build/intruder-alarm
```

On Ubuntu/Debian, use `sudo apt install libopencv-dev cmake g++`.

| Key / mouse | Action |
|---|---|
| drag left mouse | set the detection zone |
| `r` | reset the zone to the full frame |
| `d` | toggle the debug windows |
| `q` | quit |

Stay out of the picture for the first second while it shows `WARMING UP`. That's the background learning
the empty scene.

Tuning constants (learning rate, minimum threshold, minimum area, frame counts) are at the top of
[`src/main.cpp`](src/main.cpp).

## Known limits

- **Sudden lighting changes** can trigger a short false alarm before the background adapts.
- **Camera shake** shows up as motion everywhere. A fixed camera is assumed.
- A person who stays perfectly still **after** the alarm has cleared is slowly absorbed into the background.
- It detects motion, not people. A cat on the desk is an intruder too.

## How the demo GIF was made

1. Record the screen with the Fedora screen recorder (`Ctrl+Shift+Alt+R`)
2. Convert: `ffmpeg -i rec.webm -vf "fps=12,scale=640:-1" docs/demo.gif`
