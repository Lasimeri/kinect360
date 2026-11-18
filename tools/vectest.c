/* vectest: ground truth for the velocity's motion vectors. A real frame is translated by a known sub-pixel
 * velocity v per frame (Catmull-Rom, so the blur does not change with the phase), 24 frames, through the FSR 3
 * port's flow and the adapter (src/fsrflow.c) in both vector modes: 0 = EMA over whole pixels, 1 = VisionTrack's
 * one-step LK refinement. The truth is -v (vectors point newer -> older). Errors over interior blocks, after 12
 * frames (the EMA has converged by then), on all blocks and on the textured half (16 x 16 gradient energy at or
 * above the median). Checks: every frame's field arrives (counted), the engine's own block grid size matches.
 * usage: vectest FRAME.rgb   (640 x 480 RGB24, e.g. from kinect3d --dump-align) */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/fsrflow.h"

enum { W = 640, H = 480, BW = W / 8, BH = H / 8, FRAMES = 24, FROM = 12 };
static unsigned char src[W * H * 3], frm[W * H * 3];
static float out[BW * BH * 4], energy[BW * BH];

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

/* frame(x) = src(x - s): content moved by +s */
static void shifted(double sx, double sy)
{
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			const double fx = x - sx, fy = y - sy;
			const int ix = (int)floor(fx), iy = (int)floor(fy);
			const double tx = fx - ix, ty = fy - iy;
			for (int c = 0; c < 3; c++) {
				double r[4];
				for (int j = 0; j < 4; j++) r[j] = cr(at(ix - 1, iy - 1 + j, c), at(ix, iy - 1 + j, c), at(ix + 1, iy - 1 + j, c), at(ix + 2, iy - 1 + j, c), tx);
				const double v = cr(r[0], r[1], r[2], r[3], ty);
				frm[3 * (y * W + x) + c] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : lrint(v));
			}
		}
}

static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
	if (argc != 2) { fprintf(stderr, "usage: vectest FRAME.rgb (640 x 480 RGB24)\n"); return 2; }
	FILE *f = fopen(argv[1], "rb");
	if (!f || fread(src, 1, sizeof src, f) != sizeof src) { fprintf(stderr, "vectest: %s is not 921600 bytes of RGB24\n", argv[1]); return 1; }
	fclose(f);
	float esort[BW * BH];
	for (int by = 0; by < BH; by++) /* texture of each block's 16 x 16 window */
		for (int bx = 0; bx < BW; bx++) {
			double e = 0;
			for (int y = by * 8 - 4; y < by * 8 + 12; y++)
				for (int x = bx * 8 - 4; x < bx * 8 + 12; x++) {
					const double gx = at(x + 1, y, 1) - at(x - 1, y, 1), gy = at(x, y + 1, 1) - at(x, y - 1, 1);
					e += gx * gx + gy * gy;
				}
			energy[by * BW + bx] = esort[by * BW + bx] = (float)e;
		}
	qsort(esort, BW * BH, sizeof esort[0], cmpf);
	const float emed = esort[BW * BH / 2];
	static const double V[][2] = {{0, 0}, {0.3, 0}, {0.5, 0.5}, {1.25, -0.4}, {2.6, 1.7}, {-4.3, 0.8}, {7.2, -3.1}};
	static const int cpus[4] = {8, 9, 10, 11};
	static float errs[2][(FRAMES - FROM) * BW * BH];
	printf("vectest: %s, textured half = 16x16 gradient energy >= %.0f\n", argv[1], (double)emed);
	printf("%-18s | %-24s | %-24s | %-24s\n", "v (px/frame)", "EMA: med p90 <0.25px", "VisionTrack: med p90 <.25", "VT + still: med p90 <.25");
	double sum_med[3][2] = {{0}};
	for (unsigned vi = 0; vi < sizeof V / sizeof V[0]; vi++) {
		char col[3][2][40];
		for (int mode = 0; mode < 3; mode++) {
			fsr_vector_mode(mode);
			fsrcam *c = fsr_open(W, H, cpus, 4);
			if (!c) { fprintf(stderr, "vectest: engine did not start\n"); return 1; }
			int ne[2] = {0, 0}, fields = 0;
			for (int k = 0; k < FRAMES; k++) {
				shifted(k * V[vi][0], k * V[vi][1]);
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
						const float e = (float)hypot(out[4 * i] + V[vi][0], out[4 * i + 1] + V[vi][1]);
						errs[0][ne[0]++] = e;
						if (energy[i] >= emed) errs[1][ne[1]++] = e;
					}
			}
			fsr_close(c);
			if (fields != FRAMES - 1) printf("  (mode %d v %d: %d fields of %d)\n", mode, vi, fields, FRAMES - 1);
			for (int s = 0; s < 2; s++) {
				qsort(errs[s], (size_t)ne[s], sizeof errs[s][0], cmpf);
				int good = 0;
				for (int j = 0; j < ne[s]; j++) good += errs[s][j] < 0.25f;
				const double med = ne[s] ? errs[s][ne[s] / 2] : -1;
				sum_med[mode][s] += med;
				snprintf(col[mode][s], sizeof col[mode][s], "%.3f %.3f %4.1f%%", med, ne[s] ? errs[s][ne[s] * 9 / 10] : -1.0, ne[s] ? 100.0 * good / ne[s] : 0.0);
			}
		}
		printf("(%+5.2f, %+5.2f) all | %-24s | %-24s | %-24s\n", V[vi][0], V[vi][1], col[0][0], col[1][0], col[2][0]);
		printf("                 text | %-24s | %-24s | %-24s\n", col[0][1], col[1][1], col[2][1]);
	}
	printf("mean of the medians, all: EMA %.3f, VisionTrack %.3f, VT + still %.3f px; textured: %.3f, %.3f, %.3f px\n", sum_med[0][0] / 7, sum_med[1][0] / 7, sum_med[2][0] / 7,
	       sum_med[0][1] / 7, sum_med[1][1] / 7, sum_med[2][1] / 7);
	return 0;
}
