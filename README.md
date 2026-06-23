# CrossCamReID MVP

Minimal proof-of-concept for a two-camera cross-area object re-identification
system. The first MVP uses OpenCV motion detection and a lightweight visual
fingerprint, so it can run without model training.

## What This MVP Proves

- Reads two physical camera streams, or uses a built-in synthetic demo.
- Detects a moving object in each camera view.
- Tracks the object inside each camera with a local track id.
- Creates a global id for the object.
- When an object disappears from camera 1 and appears in camera 2, compares the
  new target with recently lost targets and reuses the same global id when the
  similarity is high enough.
- Draws bounding boxes, ids, similarity scores, and event logs.

This is the practical first stage of the larger plan:

```text
Camera capture -> object detection -> single-camera tracking
-> visual fingerprint extraction -> cross-camera matching -> visualization
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the synthetic demo, useful when two USB cameras are not available:

```powershell
python src/crosscam_mvp.py --demo
```

Run the same demo without a GUI window, useful for verification:

```powershell
python src/crosscam_mvp.py --demo --headless --frames 180
```

Probe local camera indexes:

```powershell
python src/crosscam_mvp.py --probe
```

Run with two physical cameras:

```powershell
python src/crosscam_mvp.py --cam-a 0 --cam-b 1
```

If the object is not detected well, improve the scene first:

- Use a static background.
- Keep lighting stable.
- Move one distinctive object at a time.
- Keep the object visible and reasonably large in the frame.
- Use a colored sticker or label on similar-looking objects.

## Recommended Project Roadmap

1. MVP: OpenCV motion detection + color/shape fingerprint matching.
2. Normal project version: train a YOLO detector for pencil/stationery objects.
3. More robust tracking: replace the local tracker with ByteTrack or BoT-SORT.
4. Better Re-ID: add ResNet, CLIP, or DINOv2 image embeddings for cropped
   object patches.
5. Stable demo fallback: use ArUco or AprilTag markers if exact identity is
   required.

## Limitations

This MVP performs probability-based matching. If two objects look exactly the
same and have no unique visible mark, a pure camera-only system cannot guarantee
perfect identity. For a high-stability demo, add a small visual marker or a
distinctive sticker.
