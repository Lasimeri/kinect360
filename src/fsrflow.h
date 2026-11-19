/* fsrflow.h: the user's FSR 3 optical flow (github.com/Lasimeri/FSR3-Optical_Flow-AVX2-FMA3, AVX2 + FMA3,
 * engine used unmodified) on one camera, with a temporal filter that turns the integer vectors' flicker
 * under sensor noise into sub-pixel motion. One 8x8 block per vector; vectors point newer -> older. */
#pragma once
#include <stdint.h>

typedef struct fsrcam fsrcam;

/* w x h luma (multiple of 8), engine threads pinned to cpus[0 .. threads-1] */
fsrcam *fsr_open(int w, int h, const int *cpus, int threads);
/* a new RGB24 frame: luma conversion and submit; returns 0, or -1 when the engine is still busy (frame skipped) */
int fsr_push(fsrcam *c, const uint8_t *rgb);
/* 1 when a new field has arrived since the last call: out = (w/8) x (h/8) x RGBA32F: filtered x, y (pixels per
 * frame; VisionTrack-refined or EMA, fsr_vector_mode), raw integer magnitude of the mid field, 1 = valid */
int fsr_poll(fsrcam *c, float *out, float *raw, uint32_t seq[2]);
/* raw (optional): (w/8) x (h/8) x 2 floats, the field as the engine made it (frame generation uses it);
 * seq (optional): the push numbers of the older and the newer frame of that field (pushes count from 1; a push
 * replaced before the engine took it is never a member, so seq[1] - seq[0] can exceed 1) */
/* block until a field is ready or CLOCK_MONOTONIC reaches deadline_s; 1 when one is ready (fsr_poll takes it) */
int fsr_wait(fsrcam *c, double deadline_s);
/* the mid-instant field alone, as soon as the flow has it (before the vectors are refined): what frame generation reads.
 * fsr_wait_mid blocks like fsr_wait; fsr_poll_mid: 1 with raw (w/8) x (h/8) x 2 floats and the field's push numbers */
int fsr_wait_mid(fsrcam *c, double deadline_s);
int fsr_poll_mid(fsrcam *c, float *raw, uint32_t seq[2]);
void fsr_stats(const fsrcam *c, unsigned *fields, unsigned *skipped, unsigned *scene_changes, double *engine_ms);
/* 0 EMA over whole pixels; 1 VisionTrack's LK refinement on moving blocks; 2 on every block; 3 through the generated frame;
 * 4 = 2 + EMA; 5 = 3 + EMA; 6 = 2 smoothed over 3 x 3 agreeing blocks and an alpha-beta filter over time (default) */
void fsr_vector_mode(int mode);
void fsr_filter_gains(float alpha, float beta); /* mode 6 */
/* blocks refined below a pixel, moving blocks that kept their whole-pixel vector, still blocks; refinement ms per field */
void fsr_refine_stats(const fsrcam *c, unsigned *refined, unsigned *kept_whole, unsigned *still, double *ms);
void fsr_close(fsrcam *c);
