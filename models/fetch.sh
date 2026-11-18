#!/bin/sh
# fetch.sh: the YuNet face detector (OpenCV Zoo, MIT licence) that kinect3d and kcalib use. Refuses a file whose
# sha256 differs from the one this project was measured with (2026-11-18: 232,589 bytes).
set -eu
D=$(cd "$(dirname "$0")" && pwd)
F=$D/face_detection_yunet_2023mar.onnx
SUM=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
URL=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
if [ ! -f "$F" ]; then curl -fsSL -o "$F.part" "$URL" && mv "$F.part" "$F"; fi
echo "$SUM  $F" | sha256sum -c -
