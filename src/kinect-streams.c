/* kinect-streams.c: isolates which Kinect stream combinations deliver frames.
 *
 * Scenarios (each SECS seconds on a freshly opened device): video RGB alone, video IR alone, video
 * Bayer alone, depth alone, depth + RGB. Per scenario: frames per stream, first-frame latency, mean
 * byte value of the last frame (black frames vs no frames), and every libfreenect log line at DEBUG,
 * deduplicated with counts. A SIGALRM watchdog guards each scenario.
 *
 * Build: tcc -Wall -Werror -I/usr/include/libfreenect -o kinect-streams kinect-streams.c -lfreenect
 * Usage: kinect-streams [SECS]          default 3
 */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <libfreenect.h>

#define NONE (-1)
typedef struct { const char *name; int depth; int video; } scen;
static const scen S[] = {
	{"rgb-alone",   0, FREENECT_VIDEO_RGB},
	{"ir-alone",    0, FREENECT_VIDEO_IR_8BIT},
	{"bayer-alone", 0, FREENECT_VIDEO_BAYER},
	{"depth-alone", 1, NONE},
	{"depth+rgb",   1, FREENECT_VIDEO_RGB},
};

static const char *volatile cur = "init";
static void on_alarm(int sig)
{
	(void)sig;
	ssize_t r = write(2, "\nTIMEOUT in scenario ", 21);
	r = write(2, (const char *)cur, strlen((const char *)cur));
	r = write(2, "\n", 1);
	(void)r;
	_exit(100);
}

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static double t_start, t_first_d, t_first_v, v_mean;
static unsigned n_depth, n_video;
static size_t v_bytes;

static void depth_cb(freenect_device *dev, void *d, uint32_t ts)
{
	(void)dev; (void)d; (void)ts;
	if (!n_depth++) t_first_d = now_s() - t_start;
}

static void video_cb(freenect_device *dev, void *v, uint32_t ts)
{
	const unsigned char *p = v;
	uint64_t s = 0;
	(void)dev; (void)ts;
	if (!n_video++) t_first_v = now_s() - t_start;
	for (size_t i = 0; i < v_bytes; i++) s += p[i];
	v_mean = v_bytes ? (double)s / (double)v_bytes : 0;
}

/* log capture: distinct lines with counts, digits folded so "Lost 3"/"Lost 4" merge */
#define MAXL 48
static char lines[MAXL][160];
static unsigned counts[MAXL], nlines, overflow;
static void log_cb(freenect_context *ctx, freenect_loglevel lvl, const char *msg)
{
	char k[160];
	size_t j = 0;
	(void)ctx;
	if (lvl > FREENECT_LOG_DEBUG) return;
	for (const char *c = msg; *c && j + 1 < sizeof k; c++) {
		if (*c == '\n' || *c == '\r') continue;
		k[j++] = (*c >= '0' && *c <= '9') ? '#' : *c;
	}
	k[j] = 0;
	for (unsigned i = 0; i < nlines; i++)
		if (!strcmp(lines[i], k)) { counts[i]++; return; }
	if (nlines < MAXL) { memcpy(lines[nlines], k, j + 1); counts[nlines++] = 1; }
	else overflow++;
}

int main(int argc, char **argv)
{
	double secs = argc > 1 ? atof(argv[1]) : 3.0;
	int bad = 0;
	if (secs <= 0) secs = 3.0;
	signal(SIGALRM, on_alarm);
	setvbuf(stdout, NULL, _IOLBF, 0);
	printf("%-12s %8s %8s %9s %9s %9s %9s  log lines\n", "scenario", "depth#", "video#", "depth fps", "video fps", "1st v (s)", "v mean");
	for (size_t si = 0; si < sizeof S / sizeof S[0]; si++) {
		const scen *sc = &S[si];
		freenect_context *ctx;
		freenect_device *dev;
		cur = sc->name;
		alarm((unsigned)secs + 10);
		if (freenect_init(&ctx, NULL) < 0) { printf("%s: freenect_init failed\n", sc->name); return 1; }
		freenect_set_log_callback(ctx, log_cb);
		freenect_set_log_level(ctx, FREENECT_LOG_DEBUG);
		freenect_select_subdevices(ctx, FREENECT_DEVICE_CAMERA);
		if (freenect_open_device(ctx, &dev, 0) < 0) { printf("%s: open failed\n", sc->name); freenect_shutdown(ctx); return 1; }
		n_depth = n_video = 0; nlines = overflow = 0; v_mean = -1; t_first_v = t_first_d = -1; v_bytes = 0;
		freenect_set_depth_callback(dev, depth_cb);
		freenect_set_video_callback(dev, video_cb);
		if (sc->video != NONE) {
			freenect_frame_mode m = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, (freenect_video_format)sc->video);
			v_bytes = (size_t)m.bytes;
			freenect_set_video_mode(dev, m);
		}
		if (sc->depth) freenect_set_depth_mode(dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
		t_start = now_s();
		if (sc->depth) freenect_start_depth(dev);
		if (sc->video != NONE) freenect_start_video(dev);
		while (now_s() - t_start < secs) {
			struct timeval tv = {0, 100000};
			if (freenect_process_events_timeout(ctx, &tv) < 0) { printf("%s: process_events error\n", sc->name); break; }
		}
		double el = now_s() - t_start;
		if (sc->video != NONE) freenect_stop_video(dev);
		if (sc->depth) freenect_stop_depth(dev);
		freenect_close_device(dev);
		freenect_shutdown(ctx);
		alarm(0);
		int want_v = sc->video != NONE, want_d = sc->depth;
		bad += (want_v && !n_video) + (want_d && !n_depth);
		printf("%-12s %8u %8u %9.2f %9.2f %9.2f %9.1f  %u distinct%s\n", sc->name, n_depth, n_video,
		       n_depth / el, n_video / el, t_first_v, v_mean, nlines, overflow ? " (+overflow)" : "");
		for (unsigned i = 0; i < nlines; i++)
			if (!strstr(lines[i], "write_register") && !strstr(lines[i], "read_register"))
				printf("      %5u x %s\n", counts[i], lines[i]);
	}
	printf("streams expected but silent: %d\n", bad);
	return bad;
}
