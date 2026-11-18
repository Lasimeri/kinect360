/* posediff: how far apart webcam calibrations put the same Kinect-frame points, by depth.
 * Reads N calib files (lines K fx fy cx cy / D k1 k2 / R 9 / t 3), projects a 21 x 16 grid over the Kinect depth
 * camera's view (f 575.82, 640 x 480) at each depth, prints median and max pixel distance of every pair. Checks:
 * every file has K, D, R, t; R orthonormal to 1e-5; a point behind a camera is counted, not projected. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { double K[4], D[2], R[9], t[3]; int have; char name[128]; } cal;
static int load(const char *p, cal *c)
{
	FILE *f = fopen(p, "r");
	char line[512];
	if (!f) { printf("cannot open %s\n", p); return -1; }
	memset(c, 0, sizeof *c);
	snprintf(c->name, sizeof c->name, "%s", strrchr(p, '/') ? strrchr(p, '/') + 1 : p);
	while (fgets(line, sizeof line, f)) {
		double *d = NULL; int n = 0, bit = 0;
		if (line[0] == 'K' && line[1] == ' ') { d = c->K; n = 4; bit = 1; }
		else if (line[0] == 'D' && line[1] == ' ') { d = c->D; n = 2; bit = 2; }
		else if (line[0] == 'R' && line[1] == ' ') { d = c->R; n = 9; bit = 4; }
		else if (line[0] == 't' && line[1] == ' ') { d = c->t; n = 3; bit = 8; }
		if (!d) continue;
		char *s = line + 2; int got = 0;
		for (; got < n; got++) { char *e; d[got] = strtod(s, &e); if (e == s) break; s = e; }
		if (got == n) c->have |= bit;
	}
	fclose(f);
	if (c->have != 15) { printf("%s: missing lines (mask %d)\n", p, c->have); return -1; }
	double worst = 0;
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
		double s = 0; for (int k = 0; k < 3; k++) s += c->R[3 * i + k] * c->R[3 * j + k];
		worst = fmax(worst, fabs(s - (i == j)));
	}
	if (worst > 1e-5) printf("%s: R not orthonormal (%.2e)\n", p, worst);
	return 0;
}
static int proj(const cal *c, const double X[3], double p[2])
{
	double q[3];
	for (int i = 0; i < 3; i++) q[i] = c->R[3 * i] * X[0] + c->R[3 * i + 1] * X[1] + c->R[3 * i + 2] * X[2] + c->t[i];
	if (q[2] <= 1.0) return 0;
	double nx = q[0] / q[2], ny = q[1] / q[2], r2 = nx * nx + ny * ny, d = 1 + r2 * (c->D[0] + r2 * c->D[1]);
	p[0] = c->K[0] * nx * d + c->K[2]; p[1] = c->K[1] * ny * d + c->K[3];
	return 1;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv)
{
	if (argc < 3) { printf("usage: posediff calib1 calib2 [...]\n"); return 2; }
	int n = argc - 1; cal *c = calloc((size_t)n, sizeof *c);
	for (int i = 0; i < n; i++) if (load(argv[i + 1], &c[i])) return 1;
	static const double Z[] = {500, 700, 1000, 1500, 2500, 3500};
	for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
		printf("%s vs %s:", c[a].name, c[b].name);
		for (unsigned zi = 0; zi < sizeof Z / sizeof Z[0]; zi++) {
			double e[400]; int m = 0, behind = 0;
			for (int gy = 0; gy < 16; gy++) for (int gx = 0; gx < 21; gx++) {
				double u = 20 + gx * 30, v = 15 + gy * 30, X[3] = {(u - 320) / 575.82 * Z[zi], (v - 240) / 575.82 * Z[zi], Z[zi]}, p[2], q[2];
				if (!proj(&c[a], X, p) || !proj(&c[b], X, q)) { behind++; continue; }
				e[m++] = hypot(p[0] - q[0], p[1] - q[1]);
			}
			qsort(e, (size_t)m, sizeof e[0], cmpd);
			printf("  %.1f m: med %.1f max %.1f px%s", Z[zi] / 1000, m ? e[m / 2] : -1.0, m ? e[m - 1] : -1.0, behind ? " (behind!)" : "");
		}
		printf("\n");
	}
	return 0;
}
