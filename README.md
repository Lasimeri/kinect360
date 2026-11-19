# kinect360

An Xbox 360 Kinect (model 1414) and a Logitech C310 webcam turned into one live, textured 3D view on Linux:
the Kinect's depth as a mesh, its colour and the webcam's mapped onto it, both cameras doubled from 30 to 60
frames a second by optical-flow frame generation, a velocity field of everything that moves in the room's own
coordinates, and a markerless face tracker in both cameras whose landmarks keep the two cameras aligned.

Everything here was built and measured on one machine: AMD Ryzen 7 5800X, NVIDIA RTX 3090 Ti (driver 615.78.08,
OpenGL 4.6 core), CachyOS (Linux 7.3), a 120 Hz monitor. Numbers below come from that machine, with the command
that produced them; what has not been measured is said to be unmeasured.

```
┌─ Kinect 1414, USB bus 3 ─────┐   ┌─ Logitech C310, /dev/video2 ──┐                         
│ libfreenect 0.7.5            │   │ V4L2 MJPEG 640x480, TurboJPEG │                         
│ depth (mm) + RGB, 30 fps     │   │ 29.9 fps                      │                         
└───┬──────────────┬───────────┘   └───────────────┬───────────────┘                         
    │ depth        │ RGB24                         │ RGB24                                   
    │              └────────────────┬──────────────┘                                         
    │                               ▼                                                        
    │      ┌──────────────────────────────────────────────────────────────────┐              
    │      │ CPU workers, one of each per camera                              │              
    │      │ FSR3 flow (cpus 0-3, 4-7): AMD field, 8x8 px blocks, 1.4-1.9 ms  │              
    │      │ + VisionTrack LK step, smoothed: 0.106 px at noise sigma 2       │              
    │      │ VisionTrack track.c, O3-native: 3.7-6.3 ms a frame; YuNet 18 ms  │              
    │      └──────┬──────────────────────┬────────────────────────┬───────────┘              
    │             │ vectors              │ mid field              │ boxes, landmarks         
    ▼             ▼                      ▼                        ▼                          
┌───────────────────────────────────────────────────────┐ ┌─────────────────────────────────┐
│ GPU: GL 4.6 compute + raster, RTX 3090 Ti             │ │ CPU main thread                 │
│ registration, occlusion z-buffer, depth filter        │ │ face aligner: landmark PnP +    │
│ velocity field in the room frame, point map           │ │ background gradient NCC -> pose │
│ frame generation 0.0075 ms: colour 30 -> 60 fps       │◀┤ room frame from accelerometer   │
│ mesh A 0.049 ms, vertices at the colour's instant     │ │ face position, velocity in room │
└──────────────────────────┬────────────────────────────┘ └─────────────────────────────────┘
                           ▼                                                                 
            ┌───────────────────────────────────────────────────────────────┐                
            │ window, 120 Hz: textured 3D view, camera panels, face boxes   │                
            └───────────────────────────────────────────────────────────────┘                
```

## Setup

```
git clone --recursive https://github.com/Lasimeri/kinect360.git    # two submodules, see Third party
cd kinect360
sh models/fetch.sh                                                   # YuNet face detector, sha256 checked
make && make check                                                   # build, gate the shaders, hardware selftest
```

Needs: libfreenect 0.7.5, GLFW, OpenGL 4.6, TurboJPEG, OpenCV 5 (core, imgproc, objdetect, dnn), glslangValidator,
tcc for the small tools. The Kinect needs its udev rules and, for its microphones, the UAC firmware:
`pkg/gen-recipes.sh` writes local variants of the two AUR recipes (libfreenect without Python, udev 0660 for the
video group with uaccess; kinect-audio-setup with a corrected udev match) from AUR clones placed beside it in `pkg/`.
Only one program can hold the Kinect at a time (close `freenect-glview` first).

## Run

```
out/kinect3d                       # the window
out/kinect3d --selftest            # hardware check: PASS/FAIL lines, exit status = FAIL count
out/kinect3d --face-stats 60       # 60 s from the first face, then the face tracker's and the aligner's report
out/kinect3d --fg-test 30          # frame generation against ground truth (below)
```

| Key | Action |
|---|---|
| left drag / right drag / wheel | orbit / pan / zoom |
| R | back to the Kinect's own viewpoint |
| O | occluded areas: Kinect colour / grey / webcam fills the gaps (default) / webcam everywhere |
| [ ] | edge cut threshold |
| F | depth filter |
| G | frame generation on/off |
| T | geometry at the colour's instant on/off |
| A | solve the webcam's pose from the still scene again |
| V | velocity arrows |
| N | camera panels: feeds (with the face boxes) / flow / off |
| M | mesh path A / B (same picture; B is the selftest's oracle) |
| W X S | tilt motor up 2 deg, down 2 deg, level |
| L, K | flow lock on/off (does not converge, off by default), save its pose |
| Esc, Q | quit |

| Option | Meaning |
|---|---|
| `--webcam DEV\|none` | webcam device (default `/dev/video2`) |
| `--calib FILE\|none` | webcam pose; default `calib/webcam-aligned.txt` if it exists, else `calib/webcam.txt` |
| `--flow-source fsr\|lk` | velocity field from the FSR 3 flow (default) or the GPU Lucas-Kanade flow |
| `--fsr-vec smooth\|lk-all\|lk\|gen\|lk-ema\|gen-ema\|ema` | the FSR flow's vectors (below); default `smooth` |
| `--no-fg`, `--no-geo-time`, `--no-face-track`, `--no-align`, `--raw-vel`, `--no-still-depth`, `--no-edge-pull`, `--no-uv-time` | turn a part off (`--raw-vel`: the cells' velocities unsmoothed) |
| `--ghost-test`, `--splat-stats N` | the depth filter on made depth; colour splatter at the live scene's silhouettes |
| `--fg-fallback` | generated pictures fall back where their two halves disagree (off: measured 0.9 dB worse) |
| `--fg-test S`, `--face-stats S`, `--scene-stats N` | measurements, then exit |
| `--dump-align DIR` | one frame of each camera, the point map and the pose, for `out/alscape` |
| `--calib-face DIR S`, `--calib-track DIR S`, `--calib-dump DIR` | calibration captures (`make calibrate-face`, `calibrate-track`, `calibrate`) |
| `--bench N`, `--method a\|b`, `--occl N`, `--cull F`, `--view YAW PITCH`, `--face-model FILE` | display and checks |
| `--lock`, `--lock-measure`, `--dump-lock DIR`, `--dump-flow DIR`, `--dump-shaders DIR`, `--analyze` | the flow lock, dumps |

## How it works

**Depth mesh.** `FREENECT_DEPTH_MM` back-projected with the device's own zero-plane calibration (reference
distance 120 mm, pixel size 0.1042 mm: f = 575.82 px). libfreenect's registration (`registration_table`,
`depth_to_rgb_shift`) runs per vertex on the GPU; a compute pass builds the RGB camera's z-buffer so surfaces it
cannot see are known. 639 x 479 quads; triangles across depth jumps are cut. A surface counts as occluded only
beyond max(50 mm, 3 depth steps), or every surface shows contour lines.

**Webcam fill.** The C310 runs MJPEG 640 x 480 at 30 fps (the only full-size mode beside the Kinect on its USB 2
bus); its pixels fill what the Kinect's colour camera cannot see, through the webcam's pose: projection
c = R X + t, focal 690.75 px (Logitech's 60 deg over the 1280 x 960 sensor).

**Optical flow and motion vectors.** Each camera feeds its own instance of the FSR 3 optical flow
([FSR3-Optical_Flow-AVX2-FMA3](https://github.com/Lasimeri/FSR3-Optical_Flow-AVX2-FMA3), AVX2 + FMA3, 4 pinned
threads each): whole-pixel vectors per 8 x 8 block. The engine's own `engine.c` keeps one flow state per process,
so `src/fsrflow.c` drives `flow_init` / `flow_land` / `flow_make` with a state per camera. Each block's vector is
then refined below a pixel by VisionTrack's one-step Lucas-Kanade (`refine_block`, from
[Intel-Phi-VisionTrack](https://github.com/Lasimeri/Intel-Phi-VisionTrack) `src/kern/fsr3.c`) against the older
frame moved by the vector, over a 16 x 16 window, on every block (VisionTrack refines moving blocks only: mode
`lk`). The default, `smooth`, then averages each vector with those of the 3 x 3 blocks that agree with it within
1 px (smoothing over space does not lag) and runs an alpha-beta filter (0.6, 0.2) over time, which follows a
steady acceleration without lag. Mode `gen` takes the vectors through the generated frame instead (two half
steps: newer to generated, generated to older); it measures no better than the direct step (below), as
VisionTrack found for its tracker ("It adds no information: everything in a generated frame comes from the two
real ones").

**Velocity smoothing.** A 3D velocity is a difference of two depth frames: one depth step at 3 m (26 mm) in 33 ms
reads as 780 mm/s whatever the image vectors say. So the cells' velocities are averaged over the 3 x 3 cells on
the same surface (depth within 10 %) and over time (EMA 0.35, restarted when a value leaves the mean by more than
twice the noise floor at that depth, so a start or a stop is not smeared). The arrows, the moving test and the
mesh's time shift read the smoothed cells.

**Frame generation.** The flow's mid-instant field moves the older frame half way forward and the newer half way
back (the convention of the flow's own `sad_mid`), bilinear between block centres; one compute dispatch per
frame. On screen: when frame k arrives, k-1 stays until the field arrives (1.4-1.9 ms, waited for up to 4 ms so it
makes the same refresh), the generated frame shows for half a period from its first refresh, then k. An image is
never replaced before it has been on screen. The mesh's vertices move by their velocity to the instant of the
colour on screen (prev, the generated midpoint, or cur), so geometry and texture refer to one moment.

**Velocity field in the room.** Kinect colour flow plus the previous frame's winner map give, per depth pixel, the
3D point it was a frame ago (sub-pixel through the registration's local slope); both points go into the room
frame (gravity from the Kinect's accelerometer, signs checked against the depth-measured ceiling normal, wall yaw
folded mod 90 deg; the same rotation for both frames unless the motor moves). Cells of 8 x 8 pixels; a cell moves
above 150 + 40 z^2 mm/s (z in m), a threshold shaped like the measured noise floor.

**Depth without ghosts: still pictures.** The depth filter follows the still pictures of VisionTrack's depth map
(`src/kern/still.c`: a running mean per camera that motion gates), on the Kinect's depth. Each pixel keeps the mean
of the frames it has held still (weight 1 / min(count, 16)); motion is judged per 8 x 8 cell: 6 of its 64 pixels
leaving their mean by more than 3 depth steps, and at least 2 such cells in a 3 x 3 (a cell with 16 changed
pixels counts as two). A moving cell's means restart from the frame itself, and nothing is held there. A pixel
that drops out keeps its mean only in a cell that is not moving, for at most 15 frames. The filter before it held
every dropped pixel for 2 frames wherever it was, which left an outline behind everything that moved.

**Colour without splatter.** Three causes, three measures:
- *The depth outline overhangs the colour outline.* Within 4.5 pixels of the end of its surface (invalid depth,
  the shadow beside every near thing, or a depth step above max(100 mm, 5 %)), a pixel's colour lookup moves
  inward until it is 4.5 px inside its own surface.
- *The colour on screen is of another instant than the depth* (up to one frame, since frame generation shows the
  previous, the generated and the newest picture in turn). Where the depth says something moved, the lookup
  follows the surface to the picture's instant along the colour's own flow.
- *The webcam's pose.* The fill paints the Kinect colour camera's blind spots from the webcam, so a wrong pose
  paints near things onto far ones. The pose is now also solved from the still scene, with no face: the gradient
  correlation of the two colour cameras through the Kinect's depth, over every point from 0.3 m (near and far
  points together separate rotation from translation), a whole-picture search then Nelder-Mead over six
  parameters, fitted on one colour of a 16 px checkerboard and judged on the other. It runs once at start (about
  0.35 s) and on the A key, and is applied when the held-out correlation rises by 0.05.

**Face tracker.** VisionTrack's region tracker (`src/kern/track.c` with its LK, Kalman and take-back, unmodified)
runs per camera on a worker thread; YuNet places its box on the face and places it again when the region is lost
or its centre leaves the face. VisionTrack's kernels are linked as one partial object (`ld -r`) whose only global
symbols are `tk_*` (`objcopy --keep-global-symbols`): 44 of its function names are also defined by the FSR 3
port. They are compiled `-O3 -march=native -ffp-contract=off`, proven bit-identical to VisionTrack's own `-O2`
build (below).

**Face alignment.** A pair of YuNet detections, one per camera, gives up to 5 correspondences: the Kinect's
landmarks as 3D points (point map), the webcam's as pixels, all carried by each tracker's rate to the instant of
the newest depth frame. PnP over R, t (Huber 3 px, Levenberg-Marquardt), judged on time blocks it never saw
(alternate 5 s blocks). Face points alone fix the pose only at the face's depth: a rotation and a translation of
Z times it cancel there. So the solve continues jointly: Nelder-Mead over all six parameters maximising the
correlation of blurred gradient magnitude between the webcam frame and the Kinect's colour carried into the
webcam, over static background points beyond 1.2 m, with the face held within 1.5 px of its best. A pose is
applied only if it beats the pose in use on held-out face error by 10 % (or ties within 1 px and wins on the
background by 0.02) and never loses on the background. It is saved to `calib/webcam-aligned.txt` at exit.

## Measured (2026-11-18)

Raw logs live in `out/` on the machine (not published: frames of the room live there too); the tools are here.

| Check | Command | Result |
|---|---|---|
| GPU colour mapping vs `freenect_map_rgb_to_depth` | `--selftest` | 0 of 307,200 pixels differ |
| mesh path A vs B, 1280 x 960 oblique render | `--selftest` | 0 of 1,228,800 pixels differ |
| GPU time per mesh draw | `--bench 600` | A 0.049 ms, B 0.079-0.081 ms |
| GPU LK flow vs OpenCV `calcOpticalFlowPyrLK` | `out/flowcheck` | median 0.0016 px |
| webcam rate after `exposure_dynamic_framerate` 0 | live | 29.88 fps (was 10.5) |

**Motion vectors, ground truth** (`out/vectest FRAME.rgb SIGMA`: a real Kinect frame moved by known sub-pixel
steps, independent Gaussian noise of SIGMA grey levels on every frame, 24 frames, median error in px over interior
blocks after 12 frames). At SIGMA 2:

| v (px/frame) | `ema` | `lk-all` | `gen` | `lk-ema` | `gen-ema` | `smooth` |
|---|---|---|---|---|---|---|
| (0, 0) | 0.000 | 0.075 | 0.087 | 0.025 | 0.027 | 0.031 |
| (0.30, 0) | 0.280 | 0.131 | 0.123 | 0.072 | 0.056 | 0.072 |
| (0.50, 0.50) | 0.248 | 0.301 | 0.329 | 0.104 | 0.216 | 0.109 |
| (1.25, -0.40) | 0.341 | 0.230 | 0.222 | 0.105 | 0.136 | 0.109 |
| (2.60, 1.70) | 0.333 | 0.248 | 0.176 | 0.109 | 0.098 | 0.113 |
| (-4.30, 0.80) | 0.314 | 0.201 | 0.252 | 0.108 | 0.159 | 0.115 |
| (7.20, -3.10) | 0.224 | 0.224 | 0.330 | 0.139 | 0.199 | 0.146 |
| sway, 6 px in 0.8 s (never constant) | 0.653 | 0.137 | 0.151 | 0.651 | 0.664 | 0.156 |
| **mean** | 0.299 | 0.193 | 0.209 | 0.164 | 0.194 | **0.106** |

| Mean of the 8 medians | `ema` | `lk-all` | `gen` | `lk-ema` | `gen-ema` | `smooth` |
|---|---|---|---|---|---|---|
| SIGMA 0 | 0.480 | 0.082 | 0.169 | 0.144 | 0.208 | **0.066** |
| SIGMA 2 | 0.299 | 0.193 | 0.209 | 0.164 | 0.194 | **0.106** |
| SIGMA 4 | 0.449 | 0.554 | 0.592 | 0.422 | 0.428 | **0.348** |

An average over time alone (`lk-ema`, `gen-ema`) halves the error at a constant velocity and lags a changing one
(the sway: 0.65 px); `smooth` does not. Vectors through the generated frame (`gen`) are within 0.02 px of the
direct ones under noise and worse without it. At 7.8 px/frame about 45 % of blocks fail in the engine's
whole-pixel search itself, before any refinement.

**Frame generation** (`--fg-test S`: a third flow instance sees every second Kinect frame; the frame generated
between k-1 and k+1 is compared with the real frame k; PSNR over pixels that move, 3 runs of 20-30 s):

| Prediction of frame k | run 1 | run 2 | run 3 |
|---|---|---|---|
| repeat k-1 | 17.94 dB | 18.67 dB | 19.47 dB |
| average of k-1 and k+1 | 21.87 dB | 22.96 dB | 23.47 dB |
| **generated** | **27.24 dB** | **26.60 dB** | **25.50 dB** |
| generated with the field's sign reversed (control) | 18.68 dB | 19.42 dB | 20.73 dB |

Generated beats the average in 346 of 451, 251 of 301 and 255 of 301 triples; 0.0075 ms of GPU a frame. On
screen: 602 Kinect frames became 1,204 images (exactly 2x), 59.98 (Kinect) and 59.81 (webcam) images a second from
30 fps cameras, no half second below 57. Display latency, loop arrival to first refresh of a real frame: 23.59 ms
(Kinect) and 27.43 ms (webcam) with generation, 4.78 and 4.02 ms without (one run each, test mode).

**Velocity field noise floor** (`--scene-stats 150`, median |v| of still cells, mm/s, two interleaved rounds):

| Depth | GPU LK flow | FSR flow (EMA vectors) |
|---|---|---|
| < 1 m | 42.9 | 25.0 / 22.9 |
| 1-2 m | 26.1 / 28.4 | 58.0 / 22.3 |
| 2-3 m | 85.1 / 57.7 | 100.7 / 106.3 |
| >= 3 m | 130.3 / 116.8 | 187.1 / 121.4 |

The FSR flow covers 4-16 times more cells and raises 3-25 times more false moving cells. Movement in the room
contaminated round 1.

With the `smooth` vectors, the cells' velocities raw (`--raw-vel`) against smoothed (two interleaved rounds, a still
room, median / 90th percentile of |v|, mm/s):

| Depth | raw | smoothed |
|---|---|---|
| 1-2 m | 9.8 / 51.6, 9.4 / 47.4 | 6.0 / 20.3, 5.4 / 21.7 |
| 2-3 m | 48.8 / 250.5, 46.6 / 236.3 | 25.4 / 124.9, 27.7 / 135.2 |
| >= 3 m | 94.3 / 328.1, 90.5 / 302.3 | 58.1 / 139.2, 58.6 / 151.7 |

**Ghosts and splatter.** `--ghost-test` runs the real depth passes on made depth with a known truth: a near
square crossing a far plane at 8 px a frame leaves 1,080.0 ghost pixels a frame behind it with the old filter and
0.0 with the still pictures; on a still plane with noise of one depth step and 5 % drop-outs the error falls from
0.578 depth steps (raw) to 0.239 (old) and 0.107 (still pictures), both with depth on 100 % of the plane.
`--splat-stats 150` asks, at every depth silhouette of the live scene, whether a pixel wears its own surface's
colour or the other's (reference colours 6 px inside each surface, only where they differ clearly; texture alone
keeps it above 0). Two interleaved rounds, old mapping against new:

| Pixels wearing the other surface's colour | at the edge | 1 px inside | 2 px inside |
|---|---|---|---|
| near surface, old | 52.9 %, 54.9 % | 41.8 %, 42.7 % | 27.4 %, 28.1 % |
| near surface, new | 10.1 %, 10.1 % | 8.9 %, 8.8 % | 7.9 %, 7.8 % |
| far surface, old | 18.5 %, 18.4 % | 13.7 %, 13.3 % | 10.7 %, 10.2 % |
| far surface, new | 6.2 %, 6.3 % | 5.2 %, 5.3 % | 4.6 %, 4.8 % |

The still-scene pose: held-out correlation 0.344 to 0.588, a change of 1.15 deg and 86 mm; the webcam-filled far
pixels in the near surface's colours fell from 57.8 % to between 18.5 and 37.8 % (it varies that much from run to
run with the scene). `out/edgeoff` on a still dump puts the colour's edges 1 to 2 px from the depth's.

**Geometry at the colour's instant** (`--fg-test 15`, with and without `--no-geo-time`): 63.0 distinct geometry
states a second on screen against 30.0, the mean shift 18.2 ms.

**Face tracker** (`--face-stats 60`, 3 runs): 3.7-7.6 ms (Kinect) and 5.4-8.4 ms (webcam) a frame, results at up to 29.6 and 28.3 Hz;
observed in 85.7-99.9 % of frames (the low end with large, fast head movement); the box's centre inside the face at 99.6-100.0 % of detections, 0.05-0.20
face widths from its centre. YuNet: 18-20 ms a run, every third frame.

**VisionTrack build** (`vt_replay`, 12 synthetic scenes x 300 frames at 640 x 480): `-O3 -march=native` against its
own `-O2` flags: 0 of 3,600 per-frame result hashes differ, rendered inputs identical, 6.40 times faster
(128,749 to 20,114 ms).

**Face alignment.** The face calibration made earlier (`make calibrate-face`) was off by 80.7 px on held-out face
landmarks by the time the aligner ran; the face-only fit reached 7.4 px, and that pose held at 5.2 px on the next
session's data, never fitted to it. But two face-only fits that agree within 20.6 px at 0.7 m disagree by 113.7 px
at 2.5 m (`out/posediff`), which is why the background term exists: on a static frame pair (`out/alscape`) the
gradient correlation has one sharp peak (0.745, the best point outside 16 px 0.583), while grey levels have none
(peak on the search boundary, margin 0.022).

## Not verified, or not working

- The joint face plus background solve has been built but not yet seen deciding live: the one run so far had no
  face in the webcam's view.
- The colour lookup's time shift is built on the measured flow but its own effect under motion is not measured (the
  rooms it ran in were nearly still); the webcam fill's remaining splatter is not separated from the two cameras'
  colour differences.
- Geometry at the colour's instant moves vertices by their cell's velocity; the depth surface itself still
  changes 30 times a second. Its picture quality has no ground truth test yet, only the count of states.
- The window's memory grew by about 78 MB an hour in one 90 minute session (2 samples of an earlier build): not
  profiled.
- The flow lock (L) does not converge: too few flow correspondences between the two cameras' colours.
- The first-open black RGB stream after the kernel's `gspca_kinect` held the camera: a restart after 3 s is
  in place; that restart has not been seen working.

## Layout

| Path | What |
|---|---|
| `src/kinect3d.c` | the program: capture, GL passes and shaders, schedule, velocity field, room, alignment |
| `src/fsrflow.[ch]` | FSR 3 flow per camera, VisionTrack's vector refinement, the mid field for generation |
| `src/vtface.[ch]` | VisionTrack's tracker per camera, kept on the face by YuNet |
| `src/flow.[ch]` | GPU pyramidal Lucas-Kanade flow (GL compute) |
| `src/webcam.[ch]`, `src/facetrack.[ch]pp` | V4L2 MJPEG capture, YuNet behind a C API |
| `src/kinect-probe.c`, `kinect-streams.c`, `v4l2-probe.c`, `holes.c`, `shiftsearch.c` | probes (tcc) |
| `tools/vectest.c` | motion vector ground truth |
| `tools/edgeoff.c` | where the colour's edges lie relative to the depth's silhouettes |
| `tools/alscape.c`, `tools/posediff.c` | the alignment's evidence: background correlation landscape, pose disagreement by depth |
| `tools/kcalib.cpp`, `tools/flowcheck.cpp`, `tools/facetest.cpp` | calibration solver, flow oracle, YuNet check |
| `tools/boxlink.c` | checks that a box-drawing diagram's connectors meet (the one above: 0 faults) |
| `models/fetch.sh` | downloads and checks the YuNet model |
| `pkg/gen-recipes.sh` | local variants of the libfreenect and kinect-audio-setup AUR recipes |
| `third_party/fsr3of`, `third_party/visiontrack` | submodules |

## Third party

- [FSR3-Optical_Flow-AVX2-FMA3](https://github.com/Lasimeri/FSR3-Optical_Flow-AVX2-FMA3) (submodule, `fb88cfd`) and
  [Intel-Phi-VisionTrack](https://github.com/Lasimeri/Intel-Phi-VisionTrack) (submodule, `ac165e8`), both used
  unmodified; their optical flow and upscaler derive from AMD's FidelityFX SDK (MIT, notices in their `LICENSES/`).
- [libfreenect](https://github.com/OpenKinect/libfreenect) (Apache 2.0 or GPL 2.0), linked, not included.
- YuNet face detector from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (MIT), downloaded by
  `models/fetch.sh`, not included.
