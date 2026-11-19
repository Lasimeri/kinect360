/* vectest: ground truth for the velocity's motion vectors. A real frame is translated by a known sub-pixel
 * velocity v per frame (Catmull-Rom, so the blur does not change with the phase), 24 frames, with independent
 * Gaussian sensor noise of SIGMA grey levels added to every frame, through the FSR 3 port's flow and the adapter
 * (src/fsrflow.c) in its six vector modes: EMA over whole pixels; VisionTrack's one-step LK on moving blocks; the
 * same on every block; through the generated frame (newer -> generated -> older); the last two with the EMA
 * over time. Seven constant velocities and one sway (never constant). The truth is -v (vectors
 * point newer -> older). Median error over interior blocks after 12 frames (the EMA has converged by then), then
 * the share of blocks under 0.25 px. Checks: every field arrives (counted); the same noise for every mode (seeded
 * per velocity and frame).
 * usage: vectest FRAME.rgb [SIGMA]   (640 x 480 RGB24, e.g. from kinect3d --dump-align; SIGMA default 0) */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/fsrflow.h"

enum { W = 640, H = 480, BW = W / 8, BH = H / 8, FRAMES = 24, FROM = 12, MODES = 7, NV = 8 };
static unsigned char src[W * H * 3], frm[W * H * 3];
static float out[BW * BH * 4];

static double cr(double p0, double p1, double p2, double p3, double t)
{
	return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

static double at(int x, int y, int c)
{
	x = x < 0 ? 0 : x >= W ? W - 1 : x;
	y = y < 0 ? 0 : y >= H ? H - 1 : y;
	return src[3 * (y * W + x) + c];
}

static uint64_t rs;
static double uni(void) /* xorshift64*, (0, 1) */
{
	rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
	return ((double)((rs * 0x2545F4914F6CDD1Dull) >> 11) + 0.5) / 9007199254740992.0;
}

/* frame(x) = src(x - s) + noise: content moved by +s; one noise value a pixel (luma noise), Box-Muller */
static void shifted(double sx, double sy, double sigma, uint64_t seed)
{
	rs = seed * 0x9E3779B97F4A7C15ull + 0x1234567ull;
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			const double fx = x - sx, fy = y - sy;
			const int ix = (int)floor(fx), iy = (int)floor(fy);
			const double tx = fx - ix, ty = fy - iy, nz = sigma > 0 ? sigma * sqrt(-2.0 * log(uni())) * cos(6.283185307179586 * uni()) : 0.0;
			for (int c = 0; c < 3; c++) {
				double r[4];
				for (int j = 0; j < 4; j++) r[j] = cr(at(ix - 1, iy - 1 + j, c), at(ix, iy - 1 + j, c), at(ix + 1, iy - 1 + j, c), at(ix + 2, iy - 1 + j, c), tx);
				const double v = cr(r[0], r[1], r[2], r[3], ty) + nz;
				frm[3 * (y * W + x) + c] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : lrint(v));
			}
		}
}

static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
	if (argc != 2 && argc != 3 && argc != 6) { fprintf(stderr, "usage: vectest FRAME.rgb [SIGMA [MODE ALPHA BETA]] (640 x 480 RGB24; the last three: one mode, with mode 6's gains)\n"); return 2; }
	const int only = argc == 6 ? atoi(argv[3]) : -1;
	if (argc == 6) fsr_filter_gains((float)atof(argv[4]), (float)atof(argv[5]));
	const double sigma = argc >= 3 ? atof(argv[2]) : 0.0;
	FILE *f = fopen(argv[1], "rb");
	if (!f || fread(src, 1, sizeof src, f) != sizeof src) { fprintf(stderr, "vectest: %s is not 921600 bytes of RGB24\n", argv[1]); return 1; }
	fclose(f);
	static const double V[NV][2] = {{0, 0}, {0.3, 0}, {0.5, 0.5}, {1.25, -0.4}, {2.6, 1.7}, {-4.3, 0.8}, {7.2, -3.1}, {0, 0}}; /* the last: a sway, below */
	static const char *const NAME[MODES] = {"EMA whole px", "VT LK moving", "VT LK all", "via generated", "VT LK all + EMA", "via gen + EMA", "VT LK space+a-b"};
	static const int cpus[4] = {8, 9, 10, 11};
	static float errs[(FRAMES - FROM) * BW * BH];
	double sum_med[MODES] = {0}, sum_good[MODES] = {0};
	printf("vectest: %s, sensor noise sigma %.1f grey levels; per mode: median error px, share under 0.25 px\n", argv[1], sigma);
	printf("%-15s", "v (px/frame)");
	for (int m = 0; m < MODES; m++)
		if (only < 0 || m == only) printf(" | %-15s", NAME[m]);
	printf("\n");
	for (int vi = 0; vi < NV; vi++) {
		if (vi == NV - 1) printf("sway 6 px 0.8 s "); else printf("(%+5.2f, %+5.2f)", V[vi][0], V[vi][1]);
		for (int mode = 0; mode < MODES; mode++) {
			if (only >= 0 && mode != only) continue;
			fsr_vector_mode(mode);
			fsrcam *c = fsr_open(W, H, cpus, 4);
			if (!c) { fprintf(stderr, "vectest: engine did not start\n"); return 1; }
			int ne = 0, fields = 0, good = 0;
			for (int k = 0; k < FRAMES; k++) {
				/* the sway: x = 6 sin(2 pi k / 24) px, a head moving side to side in 0.8 s: velocity up to 1.57 px a frame, never constant (what an average over time lags behind) */
				const int sway = vi == NV - 1;
				const double px = sway ? 6.0 * sin(6.283185307179586 * k / 24.0) : k * V[vi][0], py = sway ? 0.0 : k * V[vi][1];
				const double tvx = sway ? px - 6.0 * sin(6.283185307179586 * (k - 1) / 24.0) : V[vi][0], tvy = sway ? 0.0 : V[vi][1];
				shifted(px, py, sigma, (uint64_t)(vi * 100 + k + 1));
				fsr_push(c, frm);
				struct timespec ts;
				clock_gettime(CLOCK_MONOTONIC, &ts);
				fsr_wait(c, (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9 + 0.5);
				if (!fsr_poll(c, out, NULL, NULL)) continue;
				fields++;
				if (k < FROM) continue;
				for (int by = 2; by < BH - 2; by++)
					for (int bx = 2; bx < BW - 2; bx++) {
						const int i = by * BW + bx;
						errs[ne++] = (float)hypot(out[4 * i] + tvx, out[4 * i + 1] + tvy);
					}
			}
			fsr_close(c);
			qsort(errs, (size_t)ne, sizeof errs[0], cmpf);
			for (int j = 0; j < ne; j++) good += errs[j] < 0.25f;
			const double med = ne ? errs[ne / 2] : -1.0, gs = ne ? 100.0 * good / ne : 0.0;
			sum_med[mode] += med;
			sum_good[mode] += gs;
			printf(" | %.3f %5.1f %%%s", med, gs, fields != FRAMES - 1 ? " !" : "  ");
		}
		printf("\n");
	}
	printf("%-15s", "mean");
	for (int m = 0; m < MODES; m++)
		if (only < 0 || m == only) printf(" | %.3f %5.1f %%  ", sum_med[m] / NV, sum_good[m] / NV);
	printf("\n");
	return 0;
}
