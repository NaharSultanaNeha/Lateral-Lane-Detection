# Lateral Lane Detection & Curvature Tracking Pipeline

An end-to-end autonomous driving vision pipeline built with Python and OpenCV to detect lane boundaries, track road curvature, and compute vehicle lateral position under dynamic lighting conditions and highway curves.

---

## Key Engineering Highlights
* **Decoupled Color-Space Filtering:** Combines LAB (B-channel) for yellow lane preservation with HLS (L-channel) and directional Sobel gradients to isolate dashed white lines under intense pavement glare.
* **Camera Calibration & Rectification:** Corrects radial and tangential lens distortion using camera matrix parameters.
* **Bird’s-Eye Perspective Transform:** Uses homography matrices to map the forward-facing dashcam view into a top-down orthogonal coordinate frame.
* **Sliding Window Polynomial Fitting:** Implements iterative histogram peak detection and second-order polynomial regression ($x = Ay^2 + By + C$).
* **Parallel Curvature Coupling:** Enforces geometric consistency across lanes by coupling curvature parameters ($A, B$) of dashed markers to the continuous boundary line during temporary dropouts.
* **Real-Time Telemetry:** Computes real-world radius of road curvature and vehicle lateral lane offset in meters.

---

## Pipeline Architecture

```text
Raw Dashcam Frame
       │
       ▼
Lens Undistortion (Camera Matrix / Distortion Coefficients)
       │
       ▼
Color & Gradient Thresholding (LAB + HLS + Sobel-X)
       │
       ▼
Perspective Transform (Warp to Bird's-Eye View)
       │
       ▼
Sliding Window Histogram Search & 2nd-Order Polynomial Fit
       │
       ▼
Curvature (R_curve) & Vehicle Offset Calculation
       │
       ▼
Inverse Perspective Unwarp & HUD Telemetry Overlay
