/* fsrflow.c: see fsrflow.h. The user's FSR 3 optical flow (third_party/fsr3of, flow.c + framegen.c driven by
 * team.c, all unmodified), one instance per camera.
 *
 * Why not the project's engine.c: it keeps `static struct flow f` inside engine_main, one flow state per
 * process. With two cameras both engine threads built pyramids and fields in that one struct: level widths
 * overwritten mid-frame, `out->w == 0`, SIGFPE in fg_upscale (`idivl 0x270(%rdi)`, 1 run in 3-4, 2026-11-18).
 * So this file mirrors engine_main's sequence (team_create, flow_init FLOW_VEC, per frame flow_land then
 * flow_make, copy fg.mid) with a struct flow of its own per camera.
 *
 * Hand-off: fsr_push converts RGB to luma into the pending buffer; the worker swaps it with its work buffer
 * under the lock, so it never reads a buffer being written. A frame still pending when the next one comes is
 * replaced and counted as skipped (the field then spans two frames).
 *
 * Temporal filter, per 8x8 block: the vectors are whole pixels (framegen.c: integer search, upscale x 2); a
 * motion between two integers makes them flicker under sensor noise, and their running mean recovers the
 * fraction. m += 0.25 (v - m) while v stays within 2 px of m; a larger jump is a real change of motion and
 * resets m = v, so starts and stops are not smeared. */
#define _GNU_SOURCE
#include "fsrflow.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../third_party/fsr3of/src/kern/flow.h" /* explicit: src/flow.h (the GPU LK) shares the name */
#include "../third_party/fsr3of/src/kern/framegen.h"
#include "../third_party/fsr3of/src/engine/team.h"

/* the velocity's vectors: 1 VisionTrack's (whole-pixel field at the newer frame, refined by one LK step), 0 the
 * earlier EMA over the mid-instant field's whole pixels, 2 VisionTrack's refinement on still blocks too (sub-pixel
 * motion), 3 through the generated frame: two half steps newer -> generated -> older; frame generation always
 * reads the mid-instant field; 4 = 2 and 5 = 3 with the EMA over time (0.25, reset on a jump above 2 px);
 * 6 = 2 smoothed over space (3 x 3 blocks that agree) and by an alpha-beta filter over time */
static int g_vecmode = 6; /* ground truth (tools/vectest, sigma 2 grey levels, mean median error over 8 cases): 0 EMA 0.299, 2 LK 0.193, 3 via generated 0.209, 4 LK + EMA 0.164, 5 generated + EMA 0.194, 6 space + alpha-beta 0.106 px */
void fsr_vector_mode(int mode) { g_vecmode = mode; }
static float g_ab[2] = {0.6f, 0.2f}; /* mode 6: the alpha-beta filter's gains. vectest at sigma 2, mean of 8 cases: (0.4, 0.1) 0.125 px, (0.6, 0.2) 0.106, (1, 0) 0.129, (0.5, 0) 0.111 with the sway at 0.284 */
void fsr_filter_gains(float alpha, float beta) { g_ab[0] = alpha; g_ab[1] = beta; }

struct fsrcam {
	struct flow f __attribute__((aligned(64))); /* this camera's own flow state */
	struct team *team;
	void *mem;
	int w, h, bw, bh;
	uint8_t *luma[2]; /* [0] pending, written by fsr_push; [1] work, read by the worker */
	struct fg_vec *res;
	uint8_t *older;   /* the last frame the engine took: what refine_block steps against */
	float *ref;       /* the newer-frame field (fil[0]), refined: (w/8) x (h/8) x 2 */
	float *refw;      /* the worker's copy, swapped into ref under the lock */
	uint8_t *gen;     /* mode 3: the generated frame's luma */
	float *sp, *acc;  /* mode 6: the spatial mean, the vectors' rates */
	unsigned refined, kept_whole, still;
	double ref_ms_sum;
	pthread_t thr;
	pthread_mutex_t lock;
	pthread_cond_t wake, done; /* done: a field is ready (CLOCK_MONOTONIC, for fsr_wait) */
	int pending, quit, fresh, made, started;
	uint32_t push_seq, pend_seq, work_seq, done_seq, field_seq[2];
	float *m;
	uint8_t *ok;
	unsigned fields, skipped, scenes, runs;
	double ms_sum;
};


/* VisionTrack's motion vectors (github Lasimeri/Intel-Phi-VisionTrack ac165e8, src/kern/fsr3.c refine_block):
 * a moving block's whole-pixel vector refined below a pixel by one Lucas-Kanade step against the older frame
 * moved by that vector. Central differences of the older frame, it = newer - older(x + v), u = G^-1 sum(grad it);
 * a flat or edge-only block (det <= 1e-4 tr^2) or a step above 1.5 px keeps the whole-pixel vector. A still
 * block (0, 0) stays still (VisionTrack: its FSR reads such a block's history directly), except in mode 2, which
 * refines it too: velocity wants the sub-pixel motion the whole-pixel field calls still. Adapted: the FSR3 port's blocks are 8 px at full size, VisionTrack's 16 at half
 * size, so the window is 16 x 16 centred on the block: the same pixel support. Returns 1 if refined. */
/* One Lucas-Kanade step: the 16 x 16 template window of tmpl at (x0, y0) against ref moved by the whole-pixel
 * guess (dx, dy): tmpl(x) = ref(x + d + u). Central differences of ref, it = tmpl - ref(x + d), u = G^-1 sum(grad
 * it). 0 for a flat or edge-only window (det <= 1e-4 tr^2) or a step above 1.5 px: nothing below a pixel is known. */
static int lk_step(const uint8_t *ref, const uint8_t *tmpl, int w, int h, int x0, int y0, int dx, int dy, float *ox, float *oy)
{
	double gxx = 0, gxy = 0, gyy = 0, ex = 0, ey = 0;
	for (int r = 0; r < 16; r++) {
		const int yn = y0 + r, yo = yn + dy;
		if (yn < 0 || yn >= h || yo < 1 || yo >= h - 1) continue;
		const uint8_t *on = ref + (size_t)yo * w, *up = on - w, *dn = on + w, *nw = tmpl + (size_t)yn * w;
		for (int cc = 0; cc < 16; cc++) {
			const int xn = x0 + cc, xo = xn + dx;
			if (xn < 0 || xn >= w || xo < 1 || xo >= w - 1) continue;
			const double ix = 0.5 * ((int)on[xo + 1] - (int)on[xo - 1]), iy = 0.5 * ((int)dn[xo] - (int)up[xo]), it = (int)nw[xn] - (int)on[xo];
			gxx += ix * ix; gxy += ix * iy; gyy += iy * iy; ex += ix * it; ey += iy * it;
		}
	}
	const double det = gxx * gyy - gxy * gxy, tr = gxx + gyy;
	if (det <= 1e-4 * tr * tr || tr <= 0.0) return 0;
	const double ux = (gyy * ex - gxy * ey) / det, uy = (gxx * ey - gxy * ex) / det;
	if (fabs(ux) > 1.5 || fabs(uy) > 1.5) return 0;
	*ox = (float)(dx + ux);
	*oy = (float)(dy + uy);
	return 1;
}

static int refine_block(const uint8_t *older, const uint8_t *newer, int w, int h, int bx, int by, float *mx, float *my, int still_too)
{
	const int dx = (int)lrintf(*mx), dy = (int)lrintf(*my);
	if (!dx && !dy && !still_too) return 0;
	return lk_step(older, newer, w, h, bx * 8 - 4, by * 8 - 4, dx, dy, mx, my);
}

/* ---- vectors from the generated frame (mode 3) ----
 * gen_luma: the luma of the frame generation's picture, as fg.comp makes it: the mid-instant field bilinear between
 * block centres, the older frame at x + v/2, the newer at x - v/2, their mean. A thing at p in the newer frame and
 * p + v_true in the older lands in it at p + v_true - v/2 and at p + v/2: centred on p + v_true/2 whatever the
 * whole-pixel v was. refine_gen: two half steps through that picture, newer -> generated, generated -> older, one
 * LK step each (the window of the second follows the first), summed. Either step failing keeps the whole-pixel
 * vector. */
static inline int clampi_(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

static inline float bilin(const uint8_t *img, int w, int h, float x, float y)
{
	x = x < 0 ? 0 : x > (float)(w - 1) ? (float)(w - 1) : x;
	y = y < 0 ? 0 : y > (float)(h - 1) ? (float)(h - 1) : y;
	const int x0 = (int)x, y0 = (int)y, x1 = x0 < w - 1 ? x0 + 1 : x0, y1 = y0 < h - 1 ? y0 + 1 : y0;
	const float fx = x - (float)x0, fy = y - (float)y0;
	const uint8_t *a = img + (size_t)y0 * w, *b = img + (size_t)y1 * w;
	return (1 - fy) * ((1 - fx) * a[x0] + fx * a[x1]) + fy * ((1 - fx) * b[x0] + fx * b[x1]);
}

static void gen_luma(const struct fg_vec *mid, int bw, int bh, const uint8_t *older, const uint8_t *newer, int w, int h, uint8_t *g)
{
	for (int y = 0; y < h; y++) {
		const float fy = ((float)y + 0.5f) * 0.125f - 0.5f;
		const int j0 = clampi_((int)floorf(fy), 0, bh - 1), j1 = clampi_((int)floorf(fy) + 1, 0, bh - 1);
		const float wy = fy - floorf(fy);
		const struct fg_vec *r0 = mid + (size_t)j0 * bw, *r1 = mid + (size_t)j1 * bw;
		const uint8_t *o = older + (size_t)y * w, *n = newer + (size_t)y * w;
		uint8_t *d = g + (size_t)y * w;
		for (int x = 0; x < w; x++) {
			const float fx = ((float)x + 0.5f) * 0.125f - 0.5f;
			const int i0 = clampi_((int)floorf(fx), 0, bw - 1), i1 = clampi_((int)floorf(fx) + 1, 0, bw - 1);
			const struct fg_vec a = r0[i0], b = r0[i1], cc = r1[i0], e = r1[i1];
			if (!(a.x | a.y | b.x | b.y | cc.x | cc.y | e.x | e.y)) { d[x] = (uint8_t)((o[x] + n[x] + 1) >> 1); continue; } /* still: the mean of the two */
			const float wx = fx - floorf(fx);
			const float vx = (1 - wy) * ((1 - wx) * a.x + wx * b.x) + wy * ((1 - wx) * cc.x + wx * e.x);
			const float vy = (1 - wy) * ((1 - wx) * a.y + wx * b.y) + wy * ((1 - wx) * cc.y + wx * e.y);
			d[x] = (uint8_t)lrintf(0.5f * (bilin(older, w, h, (float)x + 0.5f * vx, (float)y + 0.5f * vy) + bilin(newer, w, h, (float)x - 0.5f * vx, (float)y - 0.5f * vy)));
		}
	}
}

static int refine_gen(const uint8_t *older, const uint8_t *gen, const uint8_t *newer, int w, int h, int bx, int by, float *mx, float *my)
{
	const int vx = (int)lrintf(*mx), vy = (int)lrintf(*my), x0 = bx * 8 - 4, y0 = by * 8 - 4;
	float ax, ay, cx, cy;
	if (!lk_step(gen, newer, w, h, x0, y0, (int)lrintf(0.5f * (float)vx), (int)lrintf(0.5f * (float)vy), &ax, &ay)) return 0;
	const int rx = (int)lrintf(ax), ry = (int)lrintf(ay);
	if (!lk_step(older, gen, w, h, x0 + rx, y0 + ry, vx - rx, vy - ry, &cx, &cy)) return 0;
	*mx = ax + cx;
	*my = ay + cy;
	return 1;
}
static void team_runner(void *ctx, flow_fn fn, void *arg) { team_run(ctx, fn, arg); }

static void *worker(void *p)
{
	fsrcam *c = p;
	const size_t vec_bytes = (size_t)c->bw * c->bh * sizeof(struct fg_vec);
	pthread_mutex_lock(&c->lock);
	for (;;) {
		while (!c->pending && !c->quit) pthread_cond_wait(&c->wake, &c->lock);
		if (c->quit) break;
		uint8_t *t = c->luma[0];
		c->luma[0] = c->luma[1];
		c->luma[1] = t;
		c->pending = 0;
		c->work_seq = c->pend_seq;
		pthread_mutex_unlock(&c->lock);
		struct timespec a, b;
		clock_gettime(CLOCK_MONOTONIC, &a);
		c->f.u8 = c->luma[1];
		c->f.u8_stride = c->w;
		flow_land(&c->f);
		const int made = flow_make(&c->f);
		clock_gettime(CLOCK_MONOTONIC, &b);
		struct timespec b2;
		const int via_gen = g_vecmode == 3 || g_vecmode == 5, all_blocks = g_vecmode == 2 || g_vecmode == 4 || g_vecmode == 6;
		if (made == 1 && via_gen) gen_luma(c->f.fg.mid.v, c->bw, c->bh, c->older, c->luma[1], c->w, c->h, c->gen);
		if (made == 1) /* VisionTrack's motion vectors from the newer-frame field, before the lock */
			for (int i = 0; i < c->bw * c->bh; i++) {
				float mx = c->f.fg.fil[0].v[i].x, my = c->f.fg.fil[0].v[i].y;
				const int r = via_gen ? refine_gen(c->older, c->gen, c->luma[1], c->w, c->h, i % c->bw, i / c->bw, &mx, &my)
							     : refine_block(c->older, c->luma[1], c->w, c->h, i % c->bw, i / c->bw, &mx, &my, all_blocks);
				c->refw[2 * i] = mx;
				c->refw[2 * i + 1] = my;
				c->refined += (unsigned)r;
				c->kept_whole += (unsigned)(!r && (mx != 0.0f || my != 0.0f));
				c->still += (unsigned)(mx == 0.0f && my == 0.0f);
			}
		memcpy(c->older, c->luma[1], (size_t)c->w * c->h);
		clock_gettime(CLOCK_MONOTONIC, &b2);
		const double ref_ms = (double)(b2.tv_sec - b.tv_sec) * 1e3 + (double)(b2.tv_nsec - b.tv_nsec) * 1e-6;
		pthread_mutex_lock(&c->lock);
		if (made == 1) { memcpy(c->res, c->f.fg.mid.v, vec_bytes); float *s = c->ref; c->ref = c->refw; c->refw = s; }
		c->made = made;
		c->field_seq[0] = c->done_seq;
		c->field_seq[1] = c->work_seq;
		c->done_seq = c->work_seq;
		c->fresh = 1;
		pthread_cond_broadcast(&c->done);
		c->ms_sum += (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
		c->runs++;
		c->ref_ms_sum += ref_ms;
	}
	pthread_mutex_unlock(&c->lock);
	return NULL;
}

void fsr_close(fsrcam *c)
{
	if (!c) return;
	if (c->started) {
		pthread_mutex_lock(&c->lock);
		c->quit = 1;
		pthread_cond_broadcast(&c->wake);
		pthread_mutex_unlock(&c->lock);
		pthread_join(c->thr, NULL);
	}
	if (c->team) team_destroy(c->team);
	free(c->mem);
	free(c->luma[0]);
	free(c->luma[1]);
	free(c->res);
	free(c->m);
	free(c->ok);
	free(c->older);
	free(c->ref);
	free(c->refw);
	free(c->gen);
	free(c->sp);
	free(c->acc);
	free(c);
}

fsrcam *fsr_open(int w, int h, const int *cpus, int threads)
{
	/* the struct holds a 64-byte aligned struct flow: calloc gives 16, and aligned AVX stores fault on that */
	const size_t bytes = (sizeof(fsrcam) + 63) & ~(size_t)63;
	fsrcam *c = aligned_alloc(64, bytes);
	if (!c) return NULL;
	memset(c, 0, bytes);
	c->w = w;
	c->h = h;
	c->bw = (w + 7) / 8;
	c->bh = (h + 7) / 8;
	const size_t fb = (flow_bytes(w, h, FLOW_VEC) + 4095) & ~(size_t)4095;
	c->mem = aligned_alloc(4096, fb);
	c->luma[0] = calloc((size_t)w * h, 1);
	c->luma[1] = calloc((size_t)w * h, 1);
	c->res = calloc((size_t)c->bw * c->bh, sizeof(struct fg_vec));
	c->m = calloc((size_t)c->bw * c->bh * 2, sizeof(float));
	c->ok = calloc((size_t)c->bw * c->bh, 1);
	c->older = calloc((size_t)w * h, 1);
	c->ref = calloc((size_t)c->bw * c->bh * 2, sizeof(float));
	c->refw = calloc((size_t)c->bw * c->bh * 2, sizeof(float));
	c->gen = calloc((size_t)w * h, 1);
	c->sp = calloc((size_t)c->bw * c->bh * 2, sizeof(float));
	c->acc = calloc((size_t)c->bw * c->bh * 2, sizeof(float));
	if (!c->mem || !c->luma[0] || !c->luma[1] || !c->res || !c->m || !c->ok || !c->older || !c->ref || !c->refw || !c->gen || !c->sp || !c->acc) { fsr_close(c); return NULL; }
	memset(c->mem, 0, fb);
	c->team = team_create(cpus, threads);
	if (!c->team || flow_init(&c->f, c->mem, w, h, FLOW_VEC, team_runner, c->team) || c->f.fg.mid.w != c->bw || c->f.fg.mid.h != c->bh || c->f.fg.fil[0].w != c->bw || c->f.fg.fil[0].h != c->bh) {
		fsr_close(c);
		return NULL;
	}
	pthread_mutex_init(&c->lock, NULL);
	pthread_cond_init(&c->wake, NULL);
	pthread_condattr_t ca;
	pthread_condattr_init(&ca);
	pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
	pthread_cond_init(&c->done, &ca);
	pthread_condattr_destroy(&ca);
	if (pthread_create(&c->thr, NULL, worker, c)) { fsr_close(c); return NULL; }
	c->started = 1;
	return c;
}

int fsr_push(fsrcam *c, const uint8_t *rgb)
{
	pthread_mutex_lock(&c->lock);
	const int dropped = c->pending;
	uint8_t *y = c->luma[0];
	for (int i = 0; i < c->w * c->h; i++, rgb += 3) y[i] = (uint8_t)((77u * rgb[0] + 150u * rgb[1] + 29u * rgb[2] + 128u) >> 8);
	c->pending = 1;
	c->pend_seq = ++c->push_seq;
	c->skipped += (unsigned)dropped;
	pthread_cond_signal(&c->wake);
	pthread_mutex_unlock(&c->lock);
	return dropped ? -1 : 0;
}

int fsr_poll(fsrcam *c, float *out, float *raw, uint32_t seq[2])
{
	pthread_mutex_lock(&c->lock);
	const int fresh = c->fresh, made = c->made;
	c->fresh = 0;
	if (fresh && made == 0) { c->scenes++; memset(c->ok, 0, (size_t)c->bw * c->bh); }
	if (fresh && made == 1) {
		const int ema = g_vecmode == 0 || g_vecmode == 4 || g_vecmode == 5; /* over time: 0 the mid field's whole pixels, 4 and 5 the refined vectors */
		const float *z = c->ref;
		if (g_vecmode == 6) { /* over space, which does not lag: the mean of the 3 x 3 blocks that agree with this one within 1 px */
			for (int by = 0; by < c->bh; by++)
				for (int bx = 0; bx < c->bw; bx++) {
					const float *me = c->ref + 2 * (by * c->bw + bx);
					float sx = 0, sy = 0;
					int n = 0;
					for (int dy = -1; dy <= 1; dy++)
						for (int dx = -1; dx <= 1; dx++) {
							const int x = bx + dx, y = by + dy;
							if (x < 0 || y < 0 || x >= c->bw || y >= c->bh) continue;
							const float *nb = c->ref + 2 * (y * c->bw + x);
							if (fabsf(nb[0] - me[0]) > 1.0f || fabsf(nb[1] - me[1]) > 1.0f) continue;
							sx += nb[0]; sy += nb[1]; n++;
						}
					c->sp[2 * (by * c->bw + bx)] = sx / (float)n;
					c->sp[2 * (by * c->bw + bx) + 1] = sy / (float)n;
				}
			z = c->sp;
		}
		for (int i = 0; i < c->bw * c->bh; i++) {
			const float mx = c->res[i].x, my = c->res[i].y;
			const float vx = g_vecmode ? z[2 * i] : mx, vy = g_vecmode ? z[2 * i + 1] : my;
			float *m = c->m + 2 * i, *a = c->acc + 2 * i;
			if (g_vecmode == 6) { /* over time without a lag under steady acceleration: an alpha-beta filter (0.4, 0.1) on the vector and its rate */
				const float px = m[0] + a[0], py = m[1] + a[1], rx = vx - px, ry = vy - py;
				if (!c->ok[i] || fabsf(rx) > 2.0f || fabsf(ry) > 2.0f) { m[0] = vx; m[1] = vy; a[0] = a[1] = 0; c->ok[i] = 1; }
				else { m[0] = px + g_ab[0] * rx; m[1] = py + g_ab[0] * ry; a[0] += g_ab[1] * rx; a[1] += g_ab[1] * ry; }
			} else if (!ema || !c->ok[i] || fabsf(vx - m[0]) > 2.0f || fabsf(vy - m[1]) > 2.0f) { m[0] = vx; m[1] = vy; c->ok[i] = 1; }
			else { m[0] += 0.25f * (vx - m[0]); m[1] += 0.25f * (vy - m[1]); }
			out[4 * i] = m[0];
			out[4 * i + 1] = m[1];
			out[4 * i + 2] = sqrtf(mx * mx + my * my);
			out[4 * i + 3] = 1.0f;
			if (raw) { raw[2 * i] = mx; raw[2 * i + 1] = my; }
		}
		c->fields++;
		if (seq) { seq[0] = c->field_seq[0]; seq[1] = c->field_seq[1]; }
	}
	pthread_mutex_unlock(&c->lock);
	return fresh && made == 1;
}

void fsr_stats(const fsrcam *c, unsigned *fields, unsigned *skipped, unsigned *scene_changes, double *engine_ms)
{
	*fields = c->fields;
	*skipped = c->skipped;
	*scene_changes = c->scenes;
	*engine_ms = c->runs ? c->ms_sum / c->runs : 0.0;
}

int fsr_wait(fsrcam *c, double deadline_s)
{
	struct timespec d;
	d.tv_sec = (time_t)deadline_s;
	d.tv_nsec = (long)((deadline_s - (double)d.tv_sec) * 1e9);
	if (d.tv_nsec >= 1000000000L) { d.tv_sec++; d.tv_nsec -= 1000000000L; }
	pthread_mutex_lock(&c->lock);
	while (!c->fresh && pthread_cond_timedwait(&c->done, &c->lock, &d) == 0) {}
	const int fresh = c->fresh;
	pthread_mutex_unlock(&c->lock);
	return fresh;
}

void fsr_refine_stats(const fsrcam *c, unsigned *refined, unsigned *kept_whole, unsigned *still, double *ms)
{
	*refined = c->refined;
	*kept_whole = c->kept_whole;
	*still = c->still;
	*ms = c->runs ? c->ref_ms_sum / c->runs : 0.0;
}
