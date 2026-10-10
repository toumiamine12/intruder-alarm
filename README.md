# Desk Intruder Alarm

<!-- ![demo](docs/demo.gif) -->

Real-time webcam motion alarm in C++17 and OpenCV. When something moves inside a user-defined detection zone,
the app shows a red **INTRUDER** banner, boxes the moving object, plays a bell and saves a timestamped snapshot.

The core pipeline (background model, histogram, Otsu threshold, binary segmentation, temporal filter) is
implemented from scratch and covered by unit tests. The Otsu implementation is verified against OpenCV's.

## Features

- Live motion detection at webcam frame rate on a standard laptop CPU
- Detection zone: drag a rectangle with the mouse, and only motion inside it counts
- Alarm: red banner, bounding box around the moving object, a bell on trigger, a snapshot every 3 s to `snapshots/`
- Debug view (`d`): difference image, binary mask, and the histogram with the selected threshold
- Noise robustness: a minimum threshold, morphological cleanup and a 5-frame confirmation
- Resizable window that scales the camera image (640×480 by default)

## Pipeline

```
webcam frame (BGR)
  → grayscale + Gaussian blur (5×5)
  → diff = |frame − background|                ← BackgroundModel::diff
  → T = Otsu threshold of diff (min 25)        ← otsuThreshold
  → binary mask = diff > T                     ← thresholdMask
  → morphological opening 3×3, closing 7×7
  → restrict to detection zone
  → contours with area ≥ 1500 px → bounding box
  → motion 5 frames in a row? → ALARM          ← MotionDebouncer
  → update background (10× slower under box)   ← BackgroundModel::update + makeSlowMask
```

## Techniques

| Technique | Where |
|---|---|
| Grayscale conversion and Gaussian smoothing | `main.cpp` |
| Histogram | `histogram()` in [`src/motion.cpp`](src/motion.cpp) |
| Automatic thresholding (Otsu's method) | `otsuThreshold()` |
| Binary segmentation | `thresholdMask()` |
| Background modelling (running average) | `BackgroundModel` |
| Morphology (opening / closing) | `main.cpp` |
| Contours and region features (area, bounding box) | `main.cpp` |
| Temporal filtering (hysteresis over frames) | `MotionDebouncer` |

## From scratch vs. OpenCV

| From scratch ([`src/motion.cpp`](src/motion.cpp)) | OpenCV |
|---|---|
| Running-average background model with a slow-learning mask | Camera capture, color → gray |
| Absolute difference image | Gaussian blur |
| 256-bin histogram + Otsu threshold | Morphological open / close |
| Thresholding to a binary mask | `findContours`, `boundingRect` |
| Motion debouncer (temporal filter) | Drawing, windows, mouse input, `imwrite` |

## Design notes

- **A still intruder isn't absorbed right away.** While the alarm is active, the background learns 10× slower
  under the bounding box, so an object that stops moving stays detected for about 20 s instead of about 2 s.
- **The alarm can't lock on.** Because the box area still learns slowly, a lasting change (an object placed on the
  desk, a lamp switched on) fades into the background and the alarm clears. If the change covers more than half
  the frame, the whole background learns at full speed and the alarm clears within a few seconds.

## Tests

`tests/test_motion.cpp` checks every from-scratch component on synthetic images, with no webcam needed:

- the histogram counts and the Otsu threshold, including uniform images and a comparison with `cv::threshold(..., THRESH_OTSU)` (within ±1 gray level)
- background learning, the slow-learning mask and re-initialization on a resolution change
- the debouncer's on/off frame counts
- an end-to-end check that the alarm clears after a lasting scene change

```bash
ctest --test-dir build --output-on-failure
```

## Build & run

**Fedora**

```bash
sudo dnf install -y opencv-devel cmake gcc-c++
```

**Ubuntu / Debian**

```bash
sudo apt install libopencv-dev cmake g++
```

**Build and run**

```bash
cmake -B build
cmake --build build
./build/intruder-alarm
```

| Key / mouse | Action |
|---|---|
| Drag with the left mouse button | Set the detection zone |
| `r` | Reset the zone to the full frame |
| `d` | Toggle the debug windows |
| `q` or close the window | Quit |

During the first second (`WARMING UP`), the scene should be empty while the background model initializes.

Tuning constants (learning rate, minimum threshold, minimum area, frame counts) are defined at the top of
[`src/main.cpp`](src/main.cpp).

## Known limitations

- **Screens in view:** monitors or TVs with changing content are detected as motion. Exclude them with the detection zone.
- **Lighting changes:** sudden changes, or objects moved within the zone, cause a false alarm. It lasts from a few
  seconds (whole-room light) up to about 25 s (a local change) until the background adapts.
- **Camera movement:** shake or repositioning registers as motion everywhere. A fixed camera is assumed.
- **Stillness:** an object that stays perfectly still after the alarm has cleared is gradually absorbed into the background.
- **Motion, not people:** it detects any movement, not specifically people.
