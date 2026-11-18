/* alscape: is the background measure of the face aligner able to judge a pose? On one static frame pair dumped by
 * kinect3d --dump-align DIR: for each calib given, NCC between the Kinect's (blurred) gradient magnitude and the
 * webcam's at the projected point, over points of the point map in a depth band; then, for the first calib, the
 * NCC over a 2D shift of the projected points (+-160 px, step 4): a measure that can judge shows one sharp peak.
 * Checks: file sizes exact; the counts of valid points printed; the zero-shift value of the landscape must equal
 * the pose's own score (control). Variants: gradient blur 2 or 6 passes; grey-level NCC as a second control. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { W = 640, H = 480 };
typedef struct { double K[4], D[2], R[9], t[3]; char name[96]; } cal;
static unsigned char kr[W * H * 3], wr[W * H * 3];
static float pm[W * H * 4], gk[W * H], gw[W * H], tmp[W * H], tmp2[W * H];
static int readall(const char *p, void *b, size_t n)
{
	FILE *f = fopen(p, "rb");
	if (!f) { printf("cannot open %s\n", p); return -1; }
	size_t g = fread(b, 1, n, f);
	int extra = fgetc(f) != EOF;
	fclose(f);
	if (g != n || extra) { printf("%s: %zu bytes read, %zu expected%s\n", p, g, n, extra ? ", file longer" : ""); return -1; }
	return 0;
}
static int load(const char *p, cal *c)
{
	FILE *f = fopen(p, "r");
	char line[512];
	int have = 0;
	if (!f) { printf("cannot open %s\n", p); return -1; }
	memset(c, 0, sizeof *c);
	snprintf(c->name, sizeof c->name, "%s", p);
	while (fgets(line, sizeof line, f)) {
		double *d = NULL; int n = 0, bit = 0;
		if (!strncmp(line, "K ", 2)) { d = c->K; n = 4; bit = 1; }
		else if (!strncmp(line, "D ", 2)) { d = c->D; n = 2; bit = 2; }
		else if (!strncmp(line, "R ", 2)) { d = c->R; n = 9; bit = 4; }
		else if (!strncmp(line, "t ", 2)) { d = c->t; n = 3; bit = 8; }
		if (!d) continue;
		char *s = line + 2; int got = 0;
		for (; got < n; got++) { char *e; d[got] = strtod(s, &e); if (e == s) break; s = e; }
		if (got == n) have |= bit;
	}
	fclose(f);
	if (have != 15) { printf("%s: incomplete (mask %d)\n", p, have); return -1; }
	return 0;
}
static void blur(float *a)
{
	for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
		const float *r = a + y * W;
		int x0 = x > 1 ? x - 2 : 0, x1 = x > 0 ? x - 1 : 0, x3 = x < W - 1 ? x + 1 : W - 1, x4 = x < W - 2 ? x + 2 : W - 1;
		tmp2[y * W + x] = (r[x0] + 4 * r[x1] + 6 * r[x] + 4 * r[x3] + r[x4]) / 16;
	}
	for (int y = 0; y < H; y++) {
		int y0 = y > 1 ? y - 2 : 0, y1 = y > 0 ? y - 1 : 0, y3 = y < H - 1 ? y + 1 : H - 1, y4 = y < H - 2 ? y + 2 : H - 1;
		for (int x = 0; x < W; x++) a[y * W + x] = (tmp2[y0 * W + x] + 4 * tmp2[y1 * W + x] + 6 * tmp2[y * W + x] + 4 * tmp2[y3 * W + x] + tmp2[y4 * W + x]) / 16;
	}
}
/* mode 0: gradient magnitude (al_grad: 2 blurs, Sobel, post blurs); mode 1: grey blurred */
static void prep(const unsigned char *rgb, float *out, int post, int mode)
{
	for (int i = 0; i < W * H; i++) tmp[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
	blur(tmp); blur(tmp);
	if (mode == 1) { memcpy(out, tmp, sizeof tmp); for (int i = 0; i < post; i++) blur(out); return; }
	for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
		if (!x || !y || x == W - 1 || y == H - 1) { out[y * W + x] = 0; continue; }
		const float *a = tmp + (y - 1) * W + x, *b = tmp + y * W + x, *c = tmp + (y + 1) * W + x;
		float gx = (a[1] + 2 * b[1] + c[1]) - (a[-1] + 2 * b[-1] + c[-1]), gy = (c[-1] + 2 * c[0] + c[1]) - (a[-1] + 2 * a[0] + a[1]);
		out[y * W + x] = sqrtf(gx * gx + gy * gy);
	}
	for (int i = 0; i < post; i++) blur(out);
}
static float *PX, *PY; /* projected position of each sampled point (or -1) */
static int NS, *SI;
static void project_all(const cal *c, double zmin)
{
	NS = 0;
	for (int y = 2; y < H - 2; y += 2) for (int x = 2; x < W - 2; x += 2) {
		const float *P = pm + 4 * (y * W + x);
		if (P[3] < 0.5f || P[2] < zmin) continue;
		double q[3];
		for (int i = 0; i < 3; i++) q[i] = c->R[3 * i] * P[0] + c->R[3 * i + 1] * P[1] + c->R[3 * i + 2] * P[2] + c->t[i];
		if (q[2] < 50) continue;
		double nx = q[0] / q[2], ny = q[1] / q[2], r2 = nx * nx + ny * ny, d = 1 + r2 * (c->D[0] + r2 * c->D[1]);
		PX[NS] = (float)(c->K[0] * nx * d + c->K[2]); PY[NS] = (float)(c->K[1] * ny * d + c->K[3]); SI[NS] = y * W + x; NS++;
	}
}
static double ncc(double du, double dv, int *cnt)
{
	double n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
	for (int i = 0; i < NS; i++) {
		double px = PX[i] + du, py = PY[i] + dv;
		if (px < 1 || py < 1 || px >= W - 2 || py >= H - 2) continue;
		int u = (int)px, v = (int)py; double fu = px - u, fv = py - v;
		const float *r0 = gw + v * W + u, *r1 = r0 + W;
		double b = (1 - fv) * ((1 - fu) * r0[0] + fu * r0[1]) + fv * ((1 - fu) * r1[0] + fu * r1[1]), a = gk[SI[i]];
		n++; sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
	}
	*cnt = (int)n;
	double den = (n * saa - sa * sa) * (n * sbb - sb * sb);
	return n > 100 && den > 0 ? (n * sab - sa * sb) / sqrt(den) : -2;
}
int main(int argc, char **argv)
{
	if (argc < 3) { printf("usage: alscape DUMPDIR calib1 [calib2 ...]\n"); return 2; }
	char p[1024];
	snprintf(p, sizeof p, "%s/kinect.rgb", argv[1]); if (readall(p, kr, sizeof kr)) return 1;
	snprintf(p, sizeof p, "%s/webcam.rgb", argv[1]); if (readall(p, wr, sizeof wr)) return 1;
	snprintf(p, sizeof p, "%s/pmap.f32", argv[1]); if (readall(p, pm, sizeof pm)) return 1;
	int nc = argc - 2; cal *c = calloc((size_t)nc, sizeof *c);
	for (int i = 0; i < nc; i++) if (load(argv[i + 2], &c[i])) return 1;
	PX = malloc(sizeof(float) * W * H); PY = malloc(sizeof(float) * W * H); SI = malloc(sizeof(int) * W * H);
	int valid = 0, far = 0;
	for (int i = 0; i < W * H; i++) { valid += pm[4 * i + 3] > 0.5f; far += pm[4 * i + 3] > 0.5f && pm[4 * i + 2] >= 1200; }
	printf("point map: %d valid of %d, %d at >= 1.2 m\n", valid, W * H, far);
	static const struct { int mode, post; const char *name; } V[] = {{0, 2, "gradient, 2 post blurs"}, {0, 6, "gradient, 6 post blurs"}, {1, 2, "grey (control)"}};
	static const double ZB[2] = {1200, 300};
	for (unsigned v = 0; v < 3; v++) {
		prep(kr, gk, V[v].post, V[v].mode); prep(wr, gw, V[v].post, V[v].mode);
		for (int zb = 0; zb < 2; zb++) {
			printf("== %s, points >= %.1f m\n", V[v].name, ZB[zb] / 1000);
			for (int k = 0; k < nc; k++) { int n; project_all(&c[k], ZB[zb]); printf("   pose %-45s NCC %.4f (%d pts)\n", c[k].name, ncc(0, 0, &n), n); }
			project_all(&c[0], ZB[zb]);
			double best = -3, bu = 0, bv = 0, z0 = -3, vals[81 * 81]; int nv = 0;
			for (int j = 0; j <= 80; j++) for (int i = 0; i <= 80; i++) {
				int n; double s = ncc(-160 + 4 * i, -160 + 4 * j, &n);
				vals[nv++] = s;
				if (i == 40 && j == 40) z0 = s;
				if (s > best) { best = s; bu = -160 + 4 * i; bv = -160 + 4 * j; }
			}
			double second = -3;
			for (int j = 0; j <= 80; j++) for (int i = 0; i <= 80; i++) {
				double du = -160 + 4 * i - bu, dv = -160 + 4 * j - bv;
				if (du * du + dv * dv > 16 * 16 && vals[j * 81 + i] > second) second = vals[j * 81 + i];
			}
			int nz; project_all(&c[0], ZB[zb]);
			printf("   shift landscape of pose 1: best %.4f at (%+.0f, %+.0f) px, best outside 16 px %.4f (margin %.4f), zero shift %.4f (control = pose 1's own %.4f)\n",
			       best, bu, bv, second, best - second, z0, ncc(0, 0, &nz));
		}
	}
	return 0;
}
