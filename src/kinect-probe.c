/* kinect-probe.c: one-pass hardware probe of an Xbox 360 Kinect (model 1414) through libfreenect.
 *
 * Stages: device count + serials, depth (mm), RGB, IR, accelerometer, LED, optional tilt round trip.
 * Controls: gravity magnitude from the accelerometer, fresh timestamps on every frame, and two
 * independent depth-statistics kernels (plain branchy loop vs branch-free histogram) that must agree
 * on every frame; both kernels are timed interleaved and the faster one is reported.
 * Every blocking stage runs under a SIGALRM watchdog that names the stage it died in.
 * Evidence frames go to OUTDIR (depth-mm.pgm 16-bit BE, depth-viz.pgm, rgb.ppm, ir.pgm).
 *
 * Build: tcc -Wall -Werror -I/usr/include/libfreenect -o kinect-probe kinect-probe.c \
 *            -lfreenect_sync -lfreenect -lm
 * Usage: kinect-probe [-n FRAMES] [-o OUTDIR] [-t]      -t moves the tilt motor +-10 deg and back
 * Exit:  number of FAIL lines (0 = every measured item passed).
 */
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <libfreenect.h>
#include <libfreenect_sync.h>

static const char *volatile stage = "init";
static int fails, undecided;

static void on_alarm(int sig)
{
	char msg[96] = "\nTIMEOUT: blocked in stage: ";
	size_t k = strlen(msg), n = strlen((const char *)stage);
	(void)sig;
	if (n > sizeof msg - k - 2) n = sizeof msg - k - 2;
	memcpy(msg + k, (const char *)stage, n);
	msg[k + n] = '\n';
	ssize_t r = write(2, msg, k + n + 1);
	(void)r;
	_exit(100);
}

static void arm(const char *name, unsigned secs) { stage = name; alarm(secs); }

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* verdict line; kind: 0 PASS, 1 FAIL, 2 UNDECIDED */
static void verdict(int kind, const char *item, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void verdict(int kind, const char *item, const char *fmt, ...)
{
	static const char *const tag[3] = {"PASS     ", "FAIL     ", "UNDECIDED"};
	va_list ap;
	fails += kind == 1;
	undecided += kind == 2;
	printf("%s %-14s ", tag[kind], item);
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	putchar('\n');
}

/* ---- depth statistics: two kernels that must agree ------------------------------------------ */
typedef struct { uint32_t valid; uint64_t sum; uint32_t min, max, med; } dstat;

static void stat_plain(const uint16_t *d, size_t n, dstat *s)
{
	uint32_t valid = 0, mn = 0xFFFF, mx = 0;
	uint64_t sum = 0;
	for (size_t i = 0; i < n; i++) {
		uint32_t v = d[i];
		if (v) {
			valid++;
			sum += v;
			if (v < mn) mn = v;
			if (v > mx) mx = v;
		}
	}
	s->valid = valid; s->sum = sum; s->min = valid ? mn : 0; s->max = mx; s->med = 0;
}

/* branch-free pixel pass: one increment per pixel into a 65,536-bin table (256 KiB, L2-resident),
 * then one scan of the bins yields valid count, sum, min, max and the median together. */
static uint32_t hist[65536];
static void stat_hist(const uint16_t *d, size_t n, dstat *s)
{
	size_t i = 0;
	memset(hist, 0, sizeof hist);
	for (; i + 4 <= n; i += 4) {
		hist[d[i]]++; hist[d[i + 1]]++; hist[d[i + 2]]++; hist[d[i + 3]]++;
	}
	for (; i < n; i++) hist[d[i]]++;
	uint32_t valid = (uint32_t)n - hist[0], mn = 0, mx = 0, med = 0, acc = 0, half = (valid + 1) / 2;
	uint64_t sum = 0;
	for (uint32_t v = 1; v < 65536; v++) {
		uint32_t c = hist[v];
		if (!c) continue;
		if (!mn) mn = v;
		mx = v;
		sum += (uint64_t)c * v;
		if (acc < half && acc + c >= half) med = v;
		acc += c;
	}
	s->valid = valid; s->sum = sum; s->min = mn; s->max = mx; s->med = med;
}

/* ---- evidence writers ----------------------------------------------------------------------- */
static int host_is_le(void) { uint16_t x = 1; unsigned char c; memcpy(&c, &x, 1); return c == 1; }

static int write_file(const char *dir, const char *name, const char *hdr, const void *p, size_t n)
{
	char path[1024];
	snprintf(path, sizeof path, "%s/%s", dir, name);
	FILE *f = fopen(path, "wb");
	if (!f) return -1;
	int ok = fputs(hdr, f) >= 0 && fwrite(p, 1, n, f) == n;
	return (fclose(f) == 0 && ok) ? 0 : -1;
}

int main(int argc, char **argv)
{
	int frames = 60, tilt = 0;
	char outdir[900] = "";
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-n") && i + 1 < argc) frames = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-o") && i + 1 < argc) snprintf(outdir, sizeof outdir, "%s", argv[++i]);
		else if (!strcmp(argv[i], "-t")) tilt = 1;
		else { fprintf(stderr, "usage: %s [-n FRAMES] [-o OUTDIR] [-t]\n", argv[0]); return 64; }
	}
	if (frames < 2) frames = 2;
	if (!outdir[0]) {
		const char *home = getenv("HOME");
		char ts[32];
		time_t t = time(NULL);
		strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", localtime(&t));
		snprintf(outdir, sizeof outdir, "%s/kinect360/out/probe-%s", home ? home : ".", ts);
	}
	if (mkdir(outdir, 0755) && errno != EEXIST) { perror(outdir); return 65; }
	signal(SIGALRM, on_alarm);
	setvbuf(stdout, NULL, _IOLBF, 0);
	printf("kinect-probe: frames=%d tilt=%d out=%s host=%s-endian\n", frames, tilt, outdir,
	       host_is_le() ? "little" : "big");

	/* ---- stage 1: enumerate through the core API -------------------------------------------- */
	freenect_context *ctx = NULL;
	arm("enumerate", 10);
	if (freenect_init(&ctx, NULL) < 0) { verdict(1, "enumerate", "freenect_init failed"); return 1; }
	freenect_set_log_level(ctx, FREENECT_LOG_WARNING);
	freenect_select_subdevices(ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));
	int nd = freenect_num_devices(ctx);
	struct freenect_device_attributes *al = NULL;
	int na = freenect_list_device_attributes(ctx, &al);
	char serial[64] = "?";
	if (al && al->camera_serial) snprintf(serial, sizeof serial, "%s", al->camera_serial);
	freenect_free_device_attributes(al);
	freenect_shutdown(ctx);
	alarm(0);
	verdict(nd >= 1 ? 0 : 1, "enumerate", "devices=%d attribute-records=%d camera-serial=%s", nd, na, serial);
	if (nd < 1) return fails;

	/* ---- stage 2: depth in mm, both kernels, timestamps --------------------------------------- */
	freenect_frame_mode dm = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM);
	if (!dm.is_valid) { verdict(1, "depth-mode", "FREENECT_DEPTH_MM medium not valid"); return fails; }
	size_t dn = (size_t)dm.width * (size_t)dm.height;
	printf("depth mode: %dx%d %d bytes\n", dm.width, dm.height, dm.bytes);
	uint16_t *keep = malloc(dn * 2);
	uint32_t prev_ts = 0, dup_ts = 0, mism = 0;
	double t0 = 0, t1 = 0, vfrac_sum = 0;
	dstat a, b;
	memset(&b, 0, sizeof b);
	for (int f = 0; f < frames; f++) {
		void *p; uint32_t ts;
		arm("depth", 10);
		if (freenect_sync_get_depth(&p, &ts, 0, FREENECT_DEPTH_MM)) { alarm(0); verdict(1, "depth", "freenect_sync_get_depth failed at frame %d", f); return fails; }
		alarm(0);
		if (f == 0) t0 = now_s();
		t1 = now_s();
		dup_ts += f && ts == prev_ts;
		prev_ts = ts;
		stat_plain(p, dn, &a);
		stat_hist(p, dn, &b);
		mism += a.valid != b.valid || a.sum != b.sum || a.min != b.min || a.max != b.max;
		vfrac_sum += (double)b.valid / (double)dn;
		if (f == frames - 1) memcpy(keep, p, dn * 2);
	}
	double dfps = (frames - 1) / (t1 - t0);
	verdict(dup_ts == 0 && dfps > 25 && dfps < 35 ? 0 : 1, "depth-stream",
		"%d frames, %.2f fps (wall), repeated timestamps=%u", frames, dfps, dup_ts);
	verdict(b.valid ? 0 : 1, "depth-content",
		"last frame: valid %u/%zu (%.1f %%), mean valid fraction %.1f %%, min %u mm, median %u mm, max %u mm, mean %.0f mm",
		b.valid, dn, 100.0 * b.valid / dn, 100.0 * vfrac_sum / frames, b.min, b.med, b.max,
		b.valid ? (double)b.sum / b.valid : 0.0);
	verdict(mism ? 1 : 0, "kernel-agree", "plain vs histogram mismatches: %u of %d frames", mism, frames);

	/* kernel timing: 7 interleaved rounds x 30 runs each on the kept frame, alternating order */
	double tp[7], th[7];
	for (int r = 0; r < 7; r++) {
		for (int k = 0; k < 2; k++) {
			int use_hist = (r + k) & 1;
			double s = now_s();
			for (int j = 0; j < 30; j++) {
				if (use_hist) stat_hist(keep, dn, &b); else stat_plain(keep, dn, &a);
			}
			double us = (now_s() - s) * 1e6 / 30;
			if (use_hist) th[r] = us; else tp[r] = us;
		}
	}
	for (int i = 0; i < 7; i++)          /* insertion sort, 7 elements each */
		for (int j = i; j > 0 && tp[j] < tp[j - 1]; j--) { double x = tp[j]; tp[j] = tp[j - 1]; tp[j - 1] = x; }
	for (int i = 0; i < 7; i++)
		for (int j = i; j > 0 && th[j] < th[j - 1]; j--) { double x = th[j]; th[j] = th[j - 1]; th[j - 1] = x; }
	printf("kernel time per %zu-pixel frame, 7 interleaved rounds x 30: plain min %.0f med %.0f max %.0f us | "
	       "histogram min %.0f med %.0f max %.0f us -> %s faster by %.2fx (median)\n",
	       dn, tp[0], tp[3], tp[6], th[0], th[3], th[6], th[3] <= tp[3] ? "histogram" : "plain",
	       th[3] <= tp[3] ? tp[3] / th[3] : th[3] / tp[3]);

	/* evidence: 16-bit big-endian PGM in mm + an 8-bit view (near = bright, no data = black) */
	{
		char hdr[64];
		uint16_t *be = malloc(dn * 2);
		unsigned char *viz = malloc(dn);
		int le = host_is_le();
		uint32_t span = b.max > b.min ? b.max - b.min : 1;
		for (size_t i = 0; i < dn; i++) {
			uint16_t v = keep[i];
			be[i] = le ? (uint16_t)((v >> 8) | (v << 8)) : v;
			viz[i] = v ? (unsigned char)(255 - (uint32_t)(v - b.min) * 223u / span) : 0;
		}
		snprintf(hdr, sizeof hdr, "P5\n%d %d\n65535\n", dm.width, dm.height);
		int e = write_file(outdir, "depth-mm.pgm", hdr, be, dn * 2);
		snprintf(hdr, sizeof hdr, "P5\n%d %d\n255\n", dm.width, dm.height);
		e |= write_file(outdir, "depth-viz.pgm", hdr, viz, dn);
		if (e) verdict(1, "depth-write", "could not write depth evidence to %s", outdir);
		free(be); free(viz);
	}

	/* ---- stage 3: RGB ------------------------------------------------------------------------- */
	freenect_frame_mode vm = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB);
	size_t vn = (size_t)vm.width * (size_t)vm.height;
	unsigned char *rgb = malloc((size_t)vm.bytes);
	dup_ts = 0;
	for (int f = 0; f < frames; f++) {
		void *p; uint32_t ts;
		arm("rgb", 10);
		if (freenect_sync_get_video(&p, &ts, 0, FREENECT_VIDEO_RGB)) { alarm(0); verdict(1, "rgb", "freenect_sync_get_video RGB failed at frame %d", f); return fails; }
		alarm(0);
		if (f == 0) t0 = now_s();
		t1 = now_s();
		dup_ts += f && ts == prev_ts;
		prev_ts = ts;
		if (f == frames - 1) memcpy(rgb, p, (size_t)vm.bytes);
	}
	double vfps = (frames - 1) / (t1 - t0);
	uint64_t ch[3] = {0, 0, 0}, sat = 0;
	for (size_t i = 0; i < vn; i++)
		for (int c = 0; c < 3; c++) { unsigned v = rgb[3 * i + c]; ch[c] += v; sat += v == 255; }
	verdict(dup_ts == 0 && vfps > 25 && vfps < 35 ? 0 : 1, "rgb-stream", "%dx%d, %d frames, %.2f fps (wall), repeated timestamps=%u",
		vm.width, vm.height, frames, vfps, dup_ts);
	verdict(ch[0] + ch[1] + ch[2] ? 0 : 1, "rgb-content", "mean R %.1f G %.1f B %.1f, saturated samples %.2f %%",
		(double)ch[0] / vn, (double)ch[1] / vn, (double)ch[2] / vn, 100.0 * sat / (3.0 * vn));
	{
		char hdr[64];
		snprintf(hdr, sizeof hdr, "P6\n%d %d\n255\n", vm.width, vm.height);
		if (write_file(outdir, "rgb.ppm", hdr, rgb, vn * 3)) verdict(1, "rgb-write", "could not write rgb.ppm");
	}

	/* ---- stage 4: IR (the projector's speckle makes the variance large) ----------------------- */
	freenect_frame_mode im = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_IR_8BIT);
	size_t in_ = (size_t)im.width * (size_t)im.height;
	{
		void *p; uint32_t ts;
		for (int f = 0; f < 10; f++) {
			arm("ir", 10);
			if (freenect_sync_get_video(&p, &ts, 0, FREENECT_VIDEO_IR_8BIT)) { alarm(0); verdict(1, "ir", "IR frame %d failed", f); return fails; }
			alarm(0);
		}
		const unsigned char *q = p;
		double s = 0, s2 = 0;
		for (size_t i = 0; i < in_; i++) { s += q[i]; s2 += (double)q[i] * q[i]; }
		double mean = s / in_, sd = sqrt(s2 / in_ - mean * mean);
		verdict(sd > 1.0 ? 0 : 1, "ir-content", "%dx%d, mean %.1f, stddev %.1f", im.width, im.height, mean, sd);
		char hdr[64];
		snprintf(hdr, sizeof hdr, "P5\n%d %d\n255\n", im.width, im.height);
		if (write_file(outdir, "ir.pgm", hdr, q, in_)) verdict(1, "ir-write", "could not write ir.pgm");
	}

	/* ---- stage 5: accelerometer: gravity is the control ---------------------------------------- */
	freenect_raw_tilt_state *st = NULL;
	double ax, ay, az;
	arm("tilt-state", 10);
	if (freenect_sync_get_tilt_state(&st, 0) || !st) { alarm(0); verdict(1, "accel", "get_tilt_state failed"); return fails; }
	alarm(0);
	freenect_get_mks_accel(st, &ax, &ay, &az);
	double g = sqrt(ax * ax + ay * ay + az * az), a0 = freenect_get_tilt_degs(st);
	verdict(fabs(g - 9.80665) <= 1.0 ? 0 : 1, "accel-gravity",
		"raw (%d,%d,%d) -> (%.2f,%.2f,%.2f) m/s^2, |g| %.3f m/s^2 (control 9.807 +-1.0), motor angle %.1f deg, status 0x%02x",
		st->accelerometer_x, st->accelerometer_y, st->accelerometer_z, ax, ay, az, g, a0, (unsigned)freenect_get_tilt_status(st));

	/* ---- stage 6: LED ------------------------------------------------------------------------- */
	arm("led", 10);
	int led = freenect_sync_set_led(LED_GREEN, 0);
	alarm(0);
	verdict(led ? 1 : 0, "led", "set_led(LED_GREEN) rc=%d (restored to BLINK_GREEN at exit)", led);

	/* ---- stage 7: tilt round trip ------------------------------------------------------------- */
	if (tilt) {
		double start = round(a0), target = start + 10 > 27 ? start - 10 : start + 10, got[2];
		double want[2] = {target, start};
		for (int leg = 0; leg < 2; leg++) {
			arm("tilt-move", 10);
			freenect_sync_set_tilt_degs((int)want[leg], 0);
			double until = now_s() + 5.0, ang = -999;
			unsigned code = 0xFF;
			usleep(300000);
			while (now_s() < until) {
				freenect_sync_get_tilt_state(&st, 0);
				code = (unsigned)freenect_get_tilt_status(st);
				ang = freenect_get_tilt_degs(st);
				if (code != TILT_STATUS_MOVING && ang > -64) break;
				usleep(100000);
			}
			alarm(0);
			got[leg] = ang;
			printf("tilt leg %d: commanded %.0f deg, settled at %.1f deg, status 0x%02x\n", leg, want[leg], ang, code);
		}
		int ok = fabs(got[0] - want[0]) <= 3.0 && fabs(got[1] - want[1]) <= 3.0;
		verdict(ok ? 0 : 1, "tilt", "start %.0f -> %.0f (got %.1f) -> %.0f (got %.1f), tolerance 3 deg",
			start, want[0], got[0], want[1], got[1]);
	} else {
		verdict(2, "tilt", "not exercised (run with -t to move the motor)");
	}

	arm("shutdown", 10);
	freenect_sync_set_led(LED_BLINK_GREEN, 0);
	freenect_sync_stop();
	alarm(0);
	printf("evidence: %s/{depth-mm.pgm,depth-viz.pgm,rgb.ppm,ir.pgm}\nsummary: %d FAIL, %d UNDECIDED\n", outdir, fails, undecided);
	free(keep); free(rgb);
	return fails;
}
