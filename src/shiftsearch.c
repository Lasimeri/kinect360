/* shiftsearch.c: global misalignment between the synthetic webcam view (lock-syn.f32, -1 = no data) and
 * the real webcam gray (lock-real.f32), 640x480 float. NCC of synthetic(p) vs real(p + d) over the
 * synthetic's valid pixels for every shift d in [-R, R]^2 (step 2, then step 1 around the best);
 * prints the best shift, its NCC, the NCC at d = 0, and the runner-up outside 10 px (peak sharpness).
 * Build: tcc -Wall -Werror -o shiftsearch shiftsearch.c -lm   Usage: shiftsearch DIR [R=120] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { W = 640, H = 480 };
static float syn[W * H], real[W * H];

static double ncc(int dx, int dy, unsigned *n_out)
{
	double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
	unsigned n = 0;
	for (int y = 0; y < H; y += 2)
		for (int x = 0; x < W; x += 2) {
			float a = syn[y * W + x];
			int u = x + dx, v = y + dy;
			if (a < 0 || u < 0 || v < 0 || u >= W || v >= H) continue;
			double b = real[v * W + u];
			sa += a; sb += b; saa += (double)a * a; sbb += b * b; sab += a * b; n++;
		}
	*n_out = n;
	if (n < 500) return -2;
	double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
	return va > 0 && vb > 0 ? cov / sqrt(va * vb) : -2;
}

static int load(const char *dir, const char *name, float *dst)
{
	char p[1024];
	snprintf(p, sizeof p, "%s/%s", dir, name);
	FILE *f = fopen(p, "rb");
	int ok = f && fread(dst, sizeof(float), W * H, f) == W * H;
	if (f) fclose(f);
	return ok ? 0 : -1;
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: %s DIR [R]\n", argv[0]); return 64; }
	int R = argc > 2 ? atoi(argv[2]) : 120;
	if (load(argv[1], "lock-syn.f32", syn) || load(argv[1], "lock-real.f32", real)) { fprintf(stderr, "cannot read the dump\n"); return 1; }
	unsigned nv = 0;
	for (int i = 0; i < W * H; i++) nv += syn[i] >= 0;
	double best = -3, second = -3;
	int bx = 0, by = 0;
	unsigned n;
	for (int dy = -R; dy <= R; dy += 2)
		for (int dx = -R; dx <= R; dx += 2) {
			double c = ncc(dx, dy, &n);
			if (c > best) { best = c; bx = dx; by = dy; }
		}
	for (int dy = by - 2; dy <= by + 2; dy++)
		for (int dx = bx - 2; dx <= bx + 2; dx++) {
			double c = ncc(dx, dy, &n);
			if (c > best) { best = c; bx = dx; by = dy; }
		}
	for (int dy = -R; dy <= R; dy += 2)
		for (int dx = -R; dx <= R; dx += 2) {
			if (abs(dx - bx) <= 10 && abs(dy - by) <= 10) continue;
			double c = ncc(dx, dy, &n);
			if (c > second) second = c;
		}
	double c0 = ncc(0, 0, &n);
	printf("synthetic valid pixels %u (%.1f %%); best shift (%+d, %+d) px NCC %.4f; at (0, 0) NCC %.4f; best outside 10 px %.4f\n", nv,
	       100.0 * nv / (W * H), bx, by, best, c0, second);
	return 0;
}
