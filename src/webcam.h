/* webcam.h: V4L2 MJPEG capture thread with TurboJPEG decode into a triple buffer (RGB24). */
#pragma once
#include <stdint.h>

typedef struct webcam webcam;

/* dev NULL: first capture node of USB 046d:081b (Logitech C310). Returns NULL with a printed reason. */
webcam *webcam_open(const char *dev, int w, int h, int fps);
/* fresh decoded frame (RGB24, w*h*3) with its kernel timestamp (CLOCK_MONOTONIC, s), or NULL */
const uint8_t *webcam_acquire(webcam *c, double *t, uint32_t *seq);
void webcam_stats(webcam *c, unsigned *frames, unsigned *fails, unsigned *gaps, double *decode_ms_mean);
const char *webcam_path(const webcam *c);
unsigned webcam_warnings(webcam *c);   /* frames decoded despite a TurboJPEG warning */
const char *webcam_error(webcam *c);   /* first decode error or warning text, "" if none */
void webcam_close(webcam *c);
