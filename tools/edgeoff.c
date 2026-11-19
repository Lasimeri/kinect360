/* edgeoff: where do the colour's edges lie relative to the depth's silhouettes, in the Kinect colour image?
 * Input: a dump of kinect3d --dump-align (kinect.rgb 640 x 480 RGB24, pmap.f32 the 3D point seen at each colour
 * pixel, RGBA32F, w = 1 valid). A silhouette: two colour pixels at most 3 apart along a row (or column), both
 * with depth, the far one beyond the near one by more than max(150 mm, 10 %), nothing valid between. For each,
 * the grey gradient along that axis at offsets -10 .. +10 px from the near surface's last pixel (positive = out
 * over the far surface), summed per offset, separately for the four facings (near surface on the left, right,
 * top, bottom). A registration that is exact peaks between offsets 0 and 1 in all four; a shift moves all four the
 * same way in image space; a depth silhouette that overhangs the colour's peaks at negative offsets in all four.
 * Checks: file sizes exact, silhouettes counted per facing, the profile printed in full. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
enum { W = 640, H = 480 };
static unsigned char rgb[W * H * 3];
static float pm[W * H * 4], grey[W * H];
static int rd(const char *dir, const char *name, void *b, size_t n)
{
	char p[1024];
	snprintf(p, sizeof p, "%s/%s", dir, name);
	FILE *f = fopen(p, "rb");
	if (!f) { printf("cannot open %s\n", p); return -1; }
	size_t g = fread(b, 1, n, f);
	int more = fgetc(f) != EOF;
	fclose(f);
	if (g != n || more) { printf("%s: %zu bytes, %zu expected%s\n", p, g, n, more ? " (longer)" : ""); return -1; }
	return 0;
}
static float z_at(int x, int y) { const float *p = pm + 4 * (y * W + x); return p[3] > 0.5f ? p[2] : 0.0f; }
int main(int argc, char **argv)
{
	if (argc != 2) { printf("usage: edgeoff DUMPDIR\n"); return 2; }
	if (rd(argv[1], "kinect.rgb", rgb, sizeof rgb) || rd(argv[1], "pmap.f32", pm, sizeof pm)) return 1;
	for (int i = 0; i < W * H; i++) grey[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
	static const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};
	static const char *const NAME[4] = {"near on the left ", "near on the right", "near on top      ", "near at bottom   "};
	double prof[4][21] = {{0}}, all[21] = {0};
	long n[4] = {0};
	for (int y = 16; y < H - 16; y++)
		for (int x = 16; x < W - 16; x++) {
			const float zn = z_at(x, y);
			if (zn <= 0) continue;
			for (int d = 0; d < 4; d++) {
				float zf = 0;
				for (int s = 1; s <= 3 && zf == 0; s++) zf = z_at(x + s * DX[d], y + s * DY[d]);
				if (zf <= zn + fmaxf(150.0f, 0.1f * zn)) continue;
				n[d]++;
				for (int o = -10; o <= 10; o++) { /* |grey step| between offsets o and o + 1 along the facing */
					const int xa = x + o * DX[d], ya = y + o * DY[d];
					prof[d][o + 10] += fabsf(grey[(ya + DY[d]) * W + xa + DX[d]] - grey[ya * W + xa]);
				}
			}
		}
	printf("edgeoff: silhouettes: %ld %ld %ld %ld; mean |grey step| between offsets o and o + 1 (o = 0: the near surface's last pixel to the next)\n", n[0], n[1], n[2], n[3]);
	printf("%-18s", "offset");
	for (int o = -10; o <= 10; o++) printf("%5d", o);
	printf("   peak\n");
	for (int d = 0; d < 4; d++) {
		int pk = 0;
		printf("%-18s", NAME[d]);
		for (int o = 0; o < 21; o++) { const double v = n[d] ? prof[d][o] / (double)n[d] : 0; all[o] += v / 4; if (prof[d][o] > prof[d][pk]) pk = o; printf("%5.1f", v); }
		printf("   %+d\n", pk - 10);
	}
	int pk = 0;
	double cen = 0, wsum = 0, base = 1e9;
	for (int o = 0; o < 21; o++) { if (all[o] > all[pk]) pk = o; if (all[o] < base) base = all[o]; }
	for (int o = 0; o < 21; o++) { cen += (all[o] - base) * (o - 10); wsum += all[o] - base; }
	printf("%-18s", "all four");
	for (int o = 0; o < 21; o++) printf("%5.1f", all[o]);
	printf("   %+d, centroid %+.2f\n", pk - 10, wsum > 0 ? cen / wsum : 0.0);
	return 0;
}
