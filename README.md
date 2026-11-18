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
    │      │ + VisionTrack LK step on every block: 0.085 px median error      │              
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
| `--fsr-vec lk-all\|lk\|ema` | the FSR flow's vectors: VisionTrack's LK refinement on every block (default), on moving blocks only, or an EMA over whole pixels |
| `--no-fg`, `--no-geo-time`, `--no-face-track`, `--no-align` | turn a part off |
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
frame moved by the vector, over a 16 x 16 window; mode `lk-all` also refines blocks the field calls still.

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

**Motion vectors, ground truth** (`out/vectest FRAME.rgb`: a real frame moved by known sub-pixel steps, 24
frames, error over interior blocks after 12 frames; median error in px; Kinect frame, the webcam frame agrees):

| v (px/frame) | EMA over whole pixels | VisionTrack LK | VisionTrack LK, still blocks too |
|---|---|---|---|
| (0.30, 0) | 0.300 | 0.300 | **0.073** |
| (0.50, 0.50) | 0.707 | 0.707 | **0.074** |
| (1.25, -0.40) | 0.504 | **0.072** | **0.072** |
| (2.60, 1.70) | 0.654 | **0.087** | **0.087** |
| (-4.30, 0.80) | 0.626 | **0.069** | **0.069** |
| (7.20, -3.10) | 0.407 | 0.218 | 0.218 |
| mean of medians (Kinect / webcam frame) | 0.457 / 0.471 | 0.208 / 0.212 | **0.085 / 0.088** |

At 7.8 px/frame about 45 % of blocks fail in the engine's whole-pixel search itself (p90 7.26 px), before any
refinement.

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
contaminated round 1. Not yet repeated with VisionTrack's vectors.

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
- Geometry at the colour's instant: built and through the shader gate; not yet run on the cameras.
- VisionTrack's vectors on still blocks (the default) on real sensor noise: the ground truth is noise-free;
  the live noise floor of the three vector modes is not measured yet.
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
