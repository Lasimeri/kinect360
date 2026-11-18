/* v4l2-probe.c: steady-state measurement of a V4L2 capture mode.
 * Opens DEV, sets WxH FOURCC at FPS, streams N frames through 4 mmap buffers and reports, from the
 * kernel's buffer timestamps: first-frame latency, steady fps over frames after WARM, interval
 * min/median/max, sequence gaps (dropped frames), payload bytes (median, for MJPEG), and whether the
 * driver granted the requested mode. Probes the timestamp clock and the granted format instead of
 * assuming them; prints what it could not decide.
 * Build: tcc -Wall -Werror -o v4l2-probe v4l2-probe.c
 * Usage: v4l2-probe DEV W H FOURCC FPS [N=150] [WARM=30]
 */
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long req, void *arg)
{
	int r;
	do r = ioctl(fd, req, arg); while (r < 0 && errno == EINTR);
	return r;
}

static int cmp_d(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

static int cmp_u(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
	if (argc < 6) { fprintf(stderr, "usage: %s DEV W H FOURCC FPS [N] [WARM]\n", argv[0]); return 64; }
	int w = atoi(argv[2]), h = atoi(argv[3]), fps = atoi(argv[5]);
	int n = argc > 6 ? atoi(argv[6]) : 150, warm = argc > 7 ? atoi(argv[7]) : 30;
	if (strlen(argv[4]) != 4 || n < warm + 10) { fprintf(stderr, "bad FOURCC or N < WARM + 10\n"); return 64; }
	uint32_t fcc = v4l2_fourcc(argv[4][0], argv[4][1], argv[4][2], argv[4][3]);
	int fd = open(argv[1], O_RDWR | O_NONBLOCK);
	if (fd < 0) { perror(argv[1]); return 1; }

	struct v4l2_format f;
	memset(&f, 0, sizeof f);
	f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	f.fmt.pix.width = (uint32_t)w; f.fmt.pix.height = (uint32_t)h; f.fmt.pix.pixelformat = fcc;
	f.fmt.pix.field = V4L2_FIELD_ANY;
	if (xioctl(fd, VIDIOC_S_FMT, &f) < 0) { perror("VIDIOC_S_FMT"); return 1; }
	struct v4l2_streamparm sp;
	memset(&sp, 0, sizeof sp);
	sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	sp.parm.capture.timeperframe.numerator = 1;
	sp.parm.capture.timeperframe.denominator = (uint32_t)fps;
	if (xioctl(fd, VIDIOC_S_PARM, &sp) < 0) perror("VIDIOC_S_PARM");
	uint32_t gf = f.fmt.pix.pixelformat;
	printf("granted %ux%u %c%c%c%c, sizeimage %u, interval %u/%u s\n", f.fmt.pix.width, f.fmt.pix.height,
	       gf & 255, (gf >> 8) & 255, (gf >> 16) & 255, gf >> 24, f.fmt.pix.sizeimage,
	       sp.parm.capture.timeperframe.numerator, sp.parm.capture.timeperframe.denominator);
	int mismatch = f.fmt.pix.width != (uint32_t)w || f.fmt.pix.height != (uint32_t)h || gf != fcc;

	struct v4l2_requestbuffers rb;
	memset(&rb, 0, sizeof rb);
	rb.count = 4; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rb.memory = V4L2_MEMORY_MMAP;
	if (xioctl(fd, VIDIOC_REQBUFS, &rb) < 0 || rb.count < 2) { perror("VIDIOC_REQBUFS"); return 1; }
	for (uint32_t i = 0; i < rb.count; i++) {
		struct v4l2_buffer b;
		memset(&b, 0, sizeof b);
		b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
		if (xioctl(fd, VIDIOC_QUERYBUF, &b) < 0) { perror("VIDIOC_QUERYBUF"); return 1; }
		if (mmap(NULL, b.length, PROT_READ, MAP_SHARED, fd, b.m.offset) == MAP_FAILED) { perror("mmap"); return 1; }
		if (xioctl(fd, VIDIOC_QBUF, &b) < 0) { perror("VIDIOC_QBUF"); return 1; }
	}
	enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	struct timespec t_on;
	clock_gettime(CLOCK_MONOTONIC, &t_on);
	if (xioctl(fd, VIDIOC_STREAMON, &t) < 0) { printf("REFUSED: VIDIOC_STREAMON: %s\n", strerror(errno)); return 2; }

	double *ts = calloc((size_t)n, sizeof *ts);
	uint32_t *seq = calloc((size_t)n, sizeof *seq), *bytes = calloc((size_t)n, sizeof *bytes);
	uint32_t clockflag = 0;
	int got = 0;
	while (got < n) {
		struct pollfd p = {fd, POLLIN, 0};
		int pr = poll(&p, 1, 3000);
		if (pr <= 0) { printf("TIMEOUT: no frame for 3 s after %d frames\n", got); break; }
		struct v4l2_buffer b;
		memset(&b, 0, sizeof b);
		b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
		if (xioctl(fd, VIDIOC_DQBUF, &b) < 0) { if (errno == EAGAIN) continue; perror("VIDIOC_DQBUF"); break; }
		ts[got] = (double)b.timestamp.tv_sec + (double)b.timestamp.tv_usec * 1e-6;
		seq[got] = b.sequence;
		bytes[got] = b.bytesused;
		clockflag = b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK;
		got++;
		xioctl(fd, VIDIOC_QBUF, &b);
	}
	xioctl(fd, VIDIOC_STREAMOFF, &t);
	if (got < warm + 10) { printf("UNDECIDED: only %d frames\n", got); return 3; }

	double on = (double)t_on.tv_sec + (double)t_on.tv_nsec * 1e-9;
	int steady = got - warm;
	double *iv = calloc((size_t)steady, sizeof *iv);
	uint32_t gaps = 0;
	for (int i = warm; i < got; i++) {
		iv[i - warm] = (ts[i] - ts[i - 1]) * 1e3;
		gaps += seq[i] - seq[i - 1] - 1;
	}
	double span = ts[got - 1] - ts[warm - 1];
	qsort(iv, (size_t)steady, sizeof *iv, cmp_d);
	qsort(bytes, (size_t)got, sizeof *bytes, cmp_u);
	printf("timestamp clock: %s\n", clockflag == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC ? "monotonic" : "UNDECIDED (not monotonic)");
	if (clockflag == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) printf("first frame %.0f ms after STREAMON\n", (ts[0] - on) * 1e3);
	printf("steady: %d frames after %d warm-up, %.3f fps, interval min %.2f median %.2f max %.2f ms, sequence gaps %u\n",
	       steady, warm, steady / span, iv[0], iv[steady / 2], iv[steady - 1], gaps);
	printf("payload bytes: min %u median %u max %u\n", bytes[0], bytes[got / 2], bytes[got - 1]);
	if (mismatch) printf("NOTE: driver changed the requested mode\n");
	return 0;
}
