/* facetrack.h: markerless face landmarks (OpenCV YuNet: eyes, nose tip, mouth corners) behind a C API.
 * Detection only: no recognition, no identity, nothing leaves the process. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct facetrack facetrack;
/* lm: right eye, left eye, nose tip, right mouth corner, left mouth corner, (x, y) in pixels */
typedef struct { float score, box[4], lm[10]; } face5;

facetrack *ft_create(const char *model, int w, int h, int threads);
int ft_detect(facetrack *f, const uint8_t *rgb, face5 *best); /* faces found (best = highest score), 0 none, -1 error */
void ft_destroy(facetrack *f);

#ifdef __cplusplus
}
#endif
