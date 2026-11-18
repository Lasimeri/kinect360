/* holes.c: where the 8-pixel depth holes come from. Grabs N raw 11-bit frames (invalid = 2047) and N
 * millimetre frames (invalid = 0), alternating, through libfreenect's sync API; per mode: invalid
 * fraction and the histogram of invalid run lengths along rows (1..15, 16+), summed over the frames.
 * Build: tcc -Wall -Werror -I/usr/include/libfreenect -o holes holes.c -lfreenect_sync -lfreenect
 * Usage: holes [N=5] */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libfreenect.h>
#include <libfreenect_sync.h>

static void on_alarm(int s) { (void)s; ssize_t r = write(2, "TIMEOUT\n", 8); (void)r; _exit(100); }

static void runs(const uint16_t *d, uint16_t bad, unsigned h[17], unsigned *nbad)
{
	for (int y = 0; y < 480; y++)
		for (int x = 0; x < 640;) {
			if (d[y * 640 + x] != bad) { x++; continue; }
			int r = 0;
			while (x < 640 && d[y * 640 + x] == bad) { r++; x++; }
			h[r > 16 ? 16 : r]++;
			*nbad += (unsigned)r;
		}
}

int main(int argc, char **argv)
{
	int n = argc > 1 ? atoi(argv[1]) : 5;
	unsigned h[2][17], nb[2] = {0, 0};
	const freenect_depth_format fmt[2] = {FREENECT_DEPTH_11BIT, FREENECT_DEPTH_MM};
	const uint16_t bad[2] = {2047, 0};
	const char *name[2] = {"raw 11-bit (invalid 2047)", "millimetre (invalid 0)"};
	memset(h, 0, sizeof h);
	signal(SIGALRM, on_alarm);
	for (int f = 0; f < n; f++)
		for (int m = 0; m < 2; m++) {
			void *p;
			uint32_t ts;
			alarm(10);
			for (int k = 0; k < 3; k++) /* after a mode change the first frames may still be the old mode */
				if (freenect_sync_get_depth(&p, &ts, 0, fmt[m])) { printf("FAIL grab\n"); return 1; }
			alarm(0);
			runs(p, bad[m], h[m], &nb[m]);
		}
	freenect_sync_stop();
	for (int m = 0; m < 2; m++) {
		printf("%-26s invalid %5.1f %%, runs 1..15, 16+:", name[m], 100.0 * nb[m] / (n * 307200.0));
		for (int r = 1; r <= 16; r++) printf(" %u", h[m][r]);
		printf("\n");
	}
	return 0;
}
