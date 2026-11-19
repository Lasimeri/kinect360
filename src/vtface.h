/* vtface.h: the user's VisionTrack region tracker (third_party/visiontrack = github Lasimeri/Intel-Phi-VisionTrack,
 * kern/track.c and its kernels unmodified, host lane model) on one camera, its box placed on the face and kept
 * there by YuNet: a select when there is no region or it is lost, a select again when the region has drifted off
 * the face (IoU below 0.3 at two detections running). One worker thread per camera; YuNet on that thread too.
 * Detection only: no recognition, no identity, nothing leaves the process. */
#pragma once
#include <stdint.h>

typedef struct vtface vtface;

enum { VTF_NONE, VTF_NEW, VTF_OBSERVED, VTF_UNOBSERVED, VTF_LOST, VTF_DESELECTED }; /* = VisionTrack's vt_state */

typedef struct {
	int state;                     /* VTF_* */
	unsigned flags, nfeat, members, track_id, missing;
	float cx, cy, w, h;            /* the region, pixels: centre and size */
	float du, dv, rate_u, rate_v;  /* displacement this frame (observed frames only), pixels per second */
	float inliers, quality, sigma_px;
	float face[4], face_score;     /* the newest YuNet box (x, y, w, h) on this camera, score 0 when none yet */
	float lm[10];                  /* its landmarks: right eye, left eye, nose tip, right and left mouth corner (x, y) */
	double t_face;                 /* capture time of the frame it was found in */
	uint64_t face_seq;             /* detections with a face so far: a new one when this grows */
	float iou;                     /* region against that box, at the newest detection; -1 none */
	double t;                      /* capture time of the frame this result is of, s (the caller's clock) */
	uint64_t frames, detections, faces, selects, reselect_drift, reselect_lost, on_face, judged, centre_in, let_go; /* centre_in: box centre inside the YuNet box; let_go: regions dropped after 30 detections without a face */
	double centre_err;             /* sum over judged detections: |box centre - face centre| / face width */
	double track_ms, track_ms_max, face_ms; /* tk_build + tk_track per frame (mean, max); YuNet per detection (mean) */
	int nshow;                     /* features to draw: member ones first */
	float feat[64][2];
	uint8_t member[64];
} vtf_out;

/* w x h (w a multiple of 16), YuNet model path, a detection every face_every-th frame (every frame while there is no region) */
vtface *vtf_open(const char *name, int w, int h, const char *face_model, int face_every);
/* a new RGB24 frame (copied), capture time t in s; a frame still waiting is replaced */
void vtf_push(vtface *v, const uint8_t *rgb, double t);
/* 1 when a result newer than the last one taken exists: copied to *o */
int vtf_get(vtface *v, vtf_out *o);
void vtf_close(vtface *v);
