/* vtface.c: see vtface.h.
 *
 * Built against VisionTrack's own headers only: kinect3d.c never sees them (the FSR3 port's vx.h, img.h and
 * framegen.h share their names). VisionTrack's kernels are linked as one partial object whose only global
 * symbols are tk_* (Makefile: ld -r, objcopy --keep-global-symbols): its fg_*, fsr_* and img_* functions, 44 names
 * the FSR3 port also defines, are local to it.
 *
 * Configuration: VisionTrack's vt_default_config (src/host/dataset.c:118, ac165e8) for one camera, mode A
 * (no stereo, no frame generation). Its comment: fitted on synthetic scenes, a real rig must fit its own; the
 * take-back threshold 0.82 was set live (2026-11-03). */
#define _GNU_SOURCE
#include "vtface.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "facetrack.h"
#include "track.h"
#include "vt_proto.h"

#define MAX_ALLOCS 256

struct vtface {
	char name[16];
	int w, h, sets, face_every, drift_run, miss_run, started, pending, quit, fresh;
	uint64_t i;
	struct tracker *tk;
	facetrack *ft;
	uint8_t *rgb[2], *grey;
	double t_in[2];
	void *allocs[MAX_ALLOCS];
	int nallocs;
	pthread_t thr;
	pthread_mutex_t lock;
	pthread_cond_t wake;
	vtf_out work, out;
	struct vt_result r; /* 20 KB: features included */
};

static void *vt_alloc(void *ctx, size_t bytes)
{
	vtface *v = ctx;
	if (v->nallocs == MAX_ALLOCS) return NULL;
	void *p = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
	if (p) { memset(p, 0, (bytes + 63) & ~(size_t)63); v->allocs[v->nallocs++] = p; }
	return p;
}

static void vt_run(void *ctx, tk_fn fn, void *arg)
{
	(void)ctx;
	fn(arg, 0, 1);
}

static void default_config(struct vt_config *c, int w, int h)
{
	memset(c, 0, sizeof *c);
	c->kind = VT_SESSION_TRACK;
	c->width = (uint32_t)w;
	c->height = (uint32_t)h;
	c->slots = 4;
	c->tpc = 2;
	c->pipelines = 1;
	c->sigma_c = 0.1f;
	c->sigma_d = 0.01f;
	c->levels = 4;
	c->patch_w = 21;
	c->patch_h = 21;
	c->max_iter = 10;
	c->eps = 0.01f;
	c->min_eig = 0.1f;
	c->max_residual = 25.0f;
	c->fb_max = 0.5f;
	c->max_features = 128;
	c->min_dist = (float)w / 80.0f;
	c->corner_quality = 0.02f;
	c->min_features = 6;
	c->min_inliers = 0.35f;
	c->prefetch = 1;
	c->max_missing = 45;
	c->reacq_ncc = 0.82f;
	c->reacq_radius = 12;
	c->fx = c->fy = 525.0f; /* mode A: no 3D from the tracker (the Kinect's depth gives it) */
	c->cx = 0.5f * (float)w;
	c->cy = 0.5f * (float)h;
	c->min_disp = 2.0f;
	c->max_disp = (float)w / 8.0f;
	c->min_ncc = 0.6f;
	c->ambiguity = 0.85f;
	c->max_reproj = 1.0f;
	c->q_accel = 0.5f;
	c->r_floor = 0.002f;
	c->v0_sigma = 2.0f;
}

static double ms_since(const struct timespec *a)
{
	struct timespec b;
	clock_gettime(CLOCK_MONOTONIC, &b);
	return (double)(b.tv_sec - a->tv_sec) * 1e3 + (double)(b.tv_nsec - a->tv_nsec) * 1e-6;
}

static float iou(const float box[4], float cx, float cy, float w, float h)
{
	const float ax0 = box[0], ay0 = box[1], ax1 = box[0] + box[2], ay1 = box[1] + box[3];
	const float bx0 = cx - 0.5f * w, by0 = cy - 0.5f * h, bx1 = cx + 0.5f * w, by1 = cy + 0.5f * h;
	const float ix = fmaxf(0.0f, fminf(ax1, bx1) - fmaxf(ax0, bx0)), iy = fmaxf(0.0f, fminf(ay1, by1) - fmaxf(ay0, by0));
	const float in = ix * iy, un = box[2] * box[3] + w * h - in;
	return un > 0.0f ? in / un : 0.0f;
}

/* one real frame on the worker: the face, the policy, the tracker */
static void step(vtface *v, const uint8_t *rgb, double t)
{
	vtf_out *o = &v->work;
	const int n = v->w * v->h;
	for (int i = 0; i < n; i++, rgb += 3) v->grey[i] = (uint8_t)((77u * rgb[0] + 150u * rgb[1] + 29u * rgb[2] + 128u) >> 8);
	rgb -= 3 * (size_t)n;
	const int active = o->state == VTF_NEW || o->state == VTF_OBSERVED || o->state == VTF_UNOBSERVED;
	int select = 0;
	if (v->ft && (!active || v->i % (uint64_t)v->face_every == 0)) {
		struct timespec a;
		clock_gettime(CLOCK_MONOTONIC, &a);
		face5 f;
		const int nf = ft_detect(v->ft, rgb, &f);
		o->face_ms += (ms_since(&a) - o->face_ms) / (double)(++o->detections);
		o->iou = -1.0f;
		v->miss_run = nf > 0 && f.score >= 0.6f ? 0 : v->miss_run + (active ? 1 : 0); /* detections in a row that found no face while a region is held */
		if (nf > 0 && f.score >= 0.6f) {
			o->faces++;
			memcpy(o->face, f.box, sizeof o->face);
			memcpy(o->lm, f.lm, sizeof o->lm);
			o->t_face = t;
			o->face_seq = o->faces;
			o->face_score = f.score;
			if (!active) {
				select = 1;
				o->reselect_lost += o->state == VTF_LOST;
			} else {
				/* off the face = the box's centre outside YuNet's box or more than half a face width from its centre.
				 * Not IoU: the tracker sizes its box to its member features, so a box centred on the face can
				 * overlap YuNet's box by less than 0.3 (webcam, 2026-11-18: IoU >= 0.5 18.3 %, centre inside 100 %) */
				const int inside = o->cx >= o->face[0] && o->cx <= o->face[0] + o->face[2] && o->cy >= o->face[1] && o->cy <= o->face[1] + o->face[3];
				const double err = hypot(o->cx - (o->face[0] + 0.5 * o->face[2]), o->cy - (o->face[1] + 0.5 * o->face[3])) / fmax(1.0, o->face[2]);
				o->iou = iou(o->face, o->cx, o->cy, o->w, o->h);
				o->judged++;
				o->on_face += o->iou >= 0.5f;
				o->centre_in += (uint64_t)inside;
				o->centre_err += err;
				v->drift_run = !inside || err > 0.5 ? v->drift_run + 1 : 0;
				if (v->drift_run >= 2) { select = 1; o->reselect_drift++; v->drift_run = 0; }
			}
		}
	}
	struct vt_msg m;
	memset(&m, 0, sizeof m);
	m.magic = VT_MSG_MAGIC;
	m.type = VT_MSG_FRAME;
	m.dest = VT_PIPE_PYR | VT_PIPE_FLOW | VT_PIPE_EST;
	m.priority = VT_PRIO_LIVE;
	m.algorithm = VT_ALG_MVEX; /* on the host: the lane model of the vector kernels */
	m.length = (uint32_t)n;
	m.instance = v->i;
	m.acq_ns = (uint64_t)(t * 1e9);
	/* A region YuNet has not found a face in for 30 detections in a row (3 s at every third frame) is let go: the tracker
	 * holds whatever it was given, and one live session read "observed" 77 % of the time with 294 faces in 45,915 detections */
	if (!select && active && v->miss_run >= 30) { m.cmd = VT_CMD_DESELECT; v->miss_run = 0; o->let_go++; }
	if (select) {
		float x0 = fmaxf(0.0f, o->face[0]), y0 = fmaxf(0.0f, o->face[1]);
		float x1 = fminf((float)v->w, o->face[0] + o->face[2]), y1 = fminf((float)v->h, o->face[1] + o->face[3]);
		if (x1 - x0 >= 16.0f && y1 - y0 >= 16.0f) {
			m.cmd = VT_CMD_SELECT;
			m.roi[0] = x0;
			m.roi[1] = y0;
			m.roi[2] = x1 - x0;
			m.roi[3] = y1 - y0;
			o->selects++;
		}
	}
	struct timespec a;
	clock_gettime(CLOCK_MONOTONIC, &a);
	const int set = (int)(v->i % (uint64_t)v->sets);
	tk_build(v->tk, set, v->grey, NULL, &m);
	tk_track(v->tk, set, &m, &v->r);
	const double ms = ms_since(&a);
	v->i++;
	o->frames++;
	o->track_ms += (ms - o->track_ms) / (double)o->frames;
	if (ms > o->track_ms_max) o->track_ms_max = ms;
	const struct vt_result *r = &v->r;
	o->state = (int)r->state;
	o->flags = r->flags;
	o->nfeat = r->nfeat;
	o->track_id = r->track_id;
	o->missing = r->missing;
	o->cx = r->cx;
	o->cy = r->cy;
	o->w = r->w;
	o->h = r->h;
	o->du = r->du;
	o->dv = r->dv;
	o->rate_u = r->rate_u;
	o->rate_v = r->rate_v;
	o->inliers = r->inliers;
	o->quality = r->quality;
	o->sigma_px = r->sigma_px;
	o->t = t;
	o->members = 0;
	o->nshow = 0;
	for (int pass = 0; pass < 2; pass++) /* member features first */
		for (unsigned k = 0; k < r->nfeat && k < VT_MAX_FEATURES && o->nshow < 64; k++) {
			const int mem = (r->feat[k].flags & VT_FF_MEMBER) != 0;
			if (pass == 0) o->members += (unsigned)mem;
			if (mem != (pass == 0)) continue;
			o->feat[o->nshow][0] = r->feat[k].x;
			o->feat[o->nshow][1] = r->feat[k].y;
			o->member[o->nshow++] = (uint8_t)mem;
		}
}

static void *worker(void *p)
{
	vtface *v = p;
	pthread_mutex_lock(&v->lock);
	for (;;) {
		while (!v->pending && !v->quit) pthread_cond_wait(&v->wake, &v->lock);
		if (v->quit) break;
		uint8_t *s = v->rgb[0];
		v->rgb[0] = v->rgb[1];
		v->rgb[1] = s;
		const double t = v->t_in[0];
		v->pending = 0;
		pthread_mutex_unlock(&v->lock);
		step(v, v->rgb[1], t);
		pthread_mutex_lock(&v->lock);
		v->out = v->work;
		v->fresh = 1;
	}
	pthread_mutex_unlock(&v->lock);
	return NULL;
}

void vtf_close(vtface *v)
{
	if (!v) return;
	if (v->started) {
		pthread_mutex_lock(&v->lock);
		v->quit = 1;
		pthread_cond_broadcast(&v->wake);
		pthread_mutex_unlock(&v->lock);
		pthread_join(v->thr, NULL);
	}
	if (v->ft) ft_destroy(v->ft);
	for (int i = 0; i < v->nallocs; i++) free(v->allocs[i]);
	free(v->rgb[0]);
	free(v->rgb[1]);
	free(v->grey);
	free(v);
}

vtface *vtf_open(const char *name, int w, int h, const char *face_model, int face_every)
{
	if (w % 16) return NULL;
	vtface *v = calloc(1, sizeof *v);
	if (!v) return NULL;
	snprintf(v->name, sizeof v->name, "%s", name);
	v->w = w;
	v->h = h;
	v->face_every = face_every > 0 ? face_every : 1;
	v->work.state = VTF_NONE;
	v->work.iou = -1.0f;
	v->rgb[0] = malloc((size_t)w * h * 3);
	v->rgb[1] = malloc((size_t)w * h * 3);
	v->grey = aligned_alloc(64, ((size_t)w * h + 63) & ~(size_t)63);
	struct vt_config cfg;
	default_config(&cfg, w, h);
	v->tk = v->rgb[0] && v->rgb[1] && v->grey ? tk_create(&cfg, 1, 1, vt_alloc, v, vt_run, NULL, vt_run, NULL) : NULL;
	v->ft = face_model ? ft_create(face_model, w, h, 2) : NULL;
	if (!v->tk || !v->ft) {
		fprintf(stderr, "vtface %s: %s\n", name, !v->tk ? "VisionTrack tracker not created" : "YuNet not loaded");
		vtf_close(v);
		return NULL;
	}
	v->sets = tk_sets(v->tk);
	pthread_mutex_init(&v->lock, NULL);
	pthread_cond_init(&v->wake, NULL);
	if (pthread_create(&v->thr, NULL, worker, v)) { vtf_close(v); return NULL; }
	v->started = 1;
	return v;
}

void vtf_push(vtface *v, const uint8_t *rgb, double t)
{
	pthread_mutex_lock(&v->lock);
	memcpy(v->rgb[0], rgb, (size_t)v->w * v->h * 3);
	v->t_in[0] = t;
	v->pending = 1;
	pthread_cond_signal(&v->wake);
	pthread_mutex_unlock(&v->lock);
}

int vtf_get(vtface *v, vtf_out *o)
{
	pthread_mutex_lock(&v->lock);
	const int fresh = v->fresh;
	if (fresh) *o = v->out;
	v->fresh = 0;
	pthread_mutex_unlock(&v->lock);
	return fresh;
}
