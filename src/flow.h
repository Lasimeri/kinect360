/* flow.h: dense pyramidal Lucas-Kanade optical flow in GL 4.6 compute shaders, one instance per camera.
 * Result texture (RGBA32F, level 0): xy = backward flow, newest frame -> previous frame, in pixels
 * (a pixel p of the newest frame was at p + xy); z = forward-backward error (px); w = 1 if valid.
 * Uses texture units 8..11 and image units 2..3 only, so callers keep their own bindings. */
#pragma once
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

typedef struct flowcam flowcam;

flowcam *flow_create(int w, int h, const char *name);
void flow_push(flowcam *f, GLuint rgb_tex, double t);   /* new frame from an RGB8 texture */
/* flow between two R32F gray images of this size: xy = where a pixel p of a is found in b */
void flow_pair(flowcam *f, GLuint gray_a, GLuint gray_b);
int flow_ready(const flowcam *f);                       /* result valid (two frames pushed) */
GLuint flow_result(const flowcam *f);
double flow_dt(const flowcam *f);                       /* seconds between the two frames */
GLuint flow_gray(const flowcam *f, int newest);         /* level-0 gray R32F, newest or previous */
double flow_gpu_ms(const flowcam *f);                   /* running mean of the GPU time per push */
int flow_dump_shaders(const char *dir);
void flow_set(float fb_max_px, float min_eig);
