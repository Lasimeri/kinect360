/* webcam.c: V4L2 MJPEG capture with TurboJPEG decode, see webcam.h.
 * Measured basis (src/v4l2-probe.c, 2026-11-18): C310 MJPEG 640x480 at 29.88 fps beside the Kinect
 * on the same USB 2 bus; larger MJPEG modes and YUYV 640x480 are refused for bandwidth there. */
#define _GNU_SOURCE
#include "webcam.h"
#include "tribuf.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <turbojpeg.h>
#include <unistd.h>

enum { NBUF = 4 };

struct webcam {
	int fd, w, h;
	unsigned nbuf;
	void *map[NBUF];
	size_t len[NBUF];
	tjhandle tj;
	tribuf tb;
	double slot_t[3];
	uint32_t slot_seq[3];
	pthread_t thr;
	atomic_int die;
	atomic_uint frames, fails, gaps, warnings;
	atomic_uint_fast64_t decode_ns;
	atomic_int have_err;
	char err[200]; /* first decode error or warning text, written once by the capture thread */
	char path[300];
};

static int xioctl(int fd, unsigned long req, void *arg)
{
	int r;
	do r = ioctl(fd, req, arg); while (r < 0 && errno == EINTR);
	return r;
}

static int read_hex(const char *path, unsigned *v)
{
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	int ok = fscanf(f, "%x", v) == 1;
	fclose(f);
	return ok ? 0 : -1;
}

/* lowest-numbered V4L2 node with video capture whose USB device is vid:pid */
static int find_node(unsigned vid, unsigned pid, char *out, size_t n)
{
	DIR *d = opendir("/sys/class/video4linux");
	int best = -1;
	if (!d) return -1;
	for (struct dirent *e; (e = readdir(d));) {
		char p[600], dev[300];
		unsigned v, q;
		if (strncmp(e->d_name, "video", 5)) continue;
		snprintf(p, sizeof p, "/sys/class/video4linux/%s/device/../idVendor", e->d_name);
		if (read_hex(p, &v)) continue;
		snprintf(p, sizeof p, "/sys/class/video4linux/%s/device/../idProduct", e->d_name);
		if (read_hex(p, &q) || v != vid || q != pid) continue;
		snprintf(dev, sizeof dev, "/dev/%s", e->d_name);
		int fd = open(dev, O_RDWR | O_NONBLOCK), num = atoi(e->d_name + 5);
		struct v4l2_capability cap;
		if (fd < 0) continue;
		memset(&cap, 0, sizeof cap);
		int ok = xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0 && (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE);
		close(fd);
		if (ok && (best < 0 || num < best)) { best = num; snprintf(out, n, "%s", dev); }
	}
	closedir(d);
	return best < 0 ? -1 : 0;
}

static void note(webcam *c, const char *stage, const char *msg)
{
	if (!atomic_exchange(&c->have_err, 1)) snprintf(c->err, sizeof c->err, "%s: %s", stage, msg);
}

/* TurboJPEG returns -1 also for warnings (TJERR_WARNING), after decoding the whole image:
 * UVC MJPEG frames often carry bytes after the image data. Only TJERR_FATAL drops a frame. */
static int decode(webcam *c, const uint8_t *jpeg, size_t n, uint8_t *dst)
{
	if (tj3DecompressHeader(c->tj, jpeg, n) < 0) {
		note(c, "header", tj3GetErrorStr(c->tj));
		if (tj3GetErrorCode(c->tj) != TJERR_WARNING) return 0;
		atomic_fetch_add(&c->warnings, 1);
	}
	int jw = tj3Get(c->tj, TJPARAM_JPEGWIDTH), jh = tj3Get(c->tj, TJPARAM_JPEGHEIGHT);
	if (jw != c->w || jh != c->h) {
		char m[64];
		snprintf(m, sizeof m, "frame is %dx%d, expected %dx%d", jw, jh, c->w, c->h);
		note(c, "size", m);
		return 0;
	}
	if (tj3Decompress8(c->tj, jpeg, n, dst, c->w * 3, TJPF_RGB) < 0) {
		note(c, "decode", tj3GetErrorStr(c->tj));
		if (tj3GetErrorCode(c->tj) != TJERR_WARNING) return 0;
		atomic_fetch_add(&c->warnings, 1);
	}
	return 1;
}

static void *cam_thread(void *arg)
{
	webcam *c = arg;
	uint32_t last = 0;
	int have = 0;
	while (!atomic_load(&c->die)) {
		struct pollfd p = {c->fd, POLLIN, 0};
		if (poll(&p, 1, 200) <= 0) continue;
		struct v4l2_buffer b;
		memset(&b, 0, sizeof b);
		b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		b.memory = V4L2_MEMORY_MMAP;
		if (xioctl(c->fd, VIDIOC_DQBUF, &b) < 0) {
			if (errno == EAGAIN) continue;
			fprintf(stderr, "webcam: VIDIOC_DQBUF: %s\n", strerror(errno));
			break;
		}
		if (have && b.sequence != last + 1) atomic_fetch_add(&c->gaps, b.sequence - last - 1);
		last = b.sequence;
		/* read before VIDIOC_QBUF: the re-queue writes the buffer's queued state back into b */
		const double ts = (double)b.timestamp.tv_sec + (double)b.timestamp.tv_usec * 1e-6;
		const uint32_t sq = b.sequence;
		have = 1;
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		int ok = decode(c, c->map[b.index], b.bytesused, c->tb.buf[c->tb.back]);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		xioctl(c->fd, VIDIOC_QBUF, &b);
		if (!ok) { atomic_fetch_add(&c->fails, 1); continue; }
		atomic_fetch_add(&c->decode_ns, (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec)));
		c->slot_t[c->tb.back] = ts;
		c->slot_seq[c->tb.back] = sq;
		tb_publish(&c->tb);
		atomic_fetch_add(&c->frames, 1);
	}
	return NULL;
}

webcam *webcam_open(const char *dev, int w, int h, int fps)
{
	webcam *c = calloc(1, sizeof *c);
	if (!c) return NULL;
	c->w = w;
	c->h = h;
	if (dev) snprintf(c->path, sizeof c->path, "%s", dev);
	else if (find_node(0x046d, 0x081b, c->path, sizeof c->path)) {
		fprintf(stderr, "webcam: no Logitech C310 (046d:081b) capture node\n");
		free(c);
		return NULL;
	}
	if ((c->fd = open(c->path, O_RDWR | O_NONBLOCK)) < 0) { fprintf(stderr, "webcam: %s: %s\n", c->path, strerror(errno)); free(c); return NULL; }

	/* the firmware lowers the frame rate to lengthen exposure when this is 1 (seen: 67.1 ms, 10.5 fps) */
	struct v4l2_control ctl = {V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0};
	if (xioctl(c->fd, VIDIOC_G_CTRL, &ctl) == 0 && ctl.value != 0) {
		fprintf(stderr, "webcam: exposure_dynamic_framerate was %d, setting 0 (constant %d fps)\n", ctl.value, fps);
		ctl.value = 0;
		xioctl(c->fd, VIDIOC_S_CTRL, &ctl);
	}
	struct v4l2_format f;
	memset(&f, 0, sizeof f);
	f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	f.fmt.pix.width = (uint32_t)w;
	f.fmt.pix.height = (uint32_t)h;
	f.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
	f.fmt.pix.field = V4L2_FIELD_ANY;
	if (xioctl(c->fd, VIDIOC_S_FMT, &f) < 0 || f.fmt.pix.width != (uint32_t)w || f.fmt.pix.height != (uint32_t)h ||
	    f.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
		fprintf(stderr, "webcam: MJPEG %dx%d not granted\n", w, h);
		goto fail;
	}
	struct v4l2_streamparm sp;
	memset(&sp, 0, sizeof sp);
	sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	sp.parm.capture.timeperframe.numerator = 1;
	sp.parm.capture.timeperframe.denominator = (uint32_t)fps;
	xioctl(c->fd, VIDIOC_S_PARM, &sp);
	struct v4l2_requestbuffers rb;
	memset(&rb, 0, sizeof rb);
	rb.count = NBUF;
	rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	rb.memory = V4L2_MEMORY_MMAP;
	if (xioctl(c->fd, VIDIOC_REQBUFS, &rb) < 0 || rb.count < 2) { fprintf(stderr, "webcam: VIDIOC_REQBUFS failed\n"); goto fail; }
	c->nbuf = rb.count > NBUF ? NBUF : rb.count;
	for (unsigned i = 0; i < c->nbuf; i++) {
		struct v4l2_buffer b;
		memset(&b, 0, sizeof b);
		b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		b.memory = V4L2_MEMORY_MMAP;
		b.index = i;
		if (xioctl(c->fd, VIDIOC_QUERYBUF, &b) < 0) goto fail;
		c->len[i] = b.length;
		if ((c->map[i] = mmap(NULL, b.length, PROT_READ, MAP_SHARED, c->fd, b.m.offset)) == MAP_FAILED) goto fail;
		if (xioctl(c->fd, VIDIOC_QBUF, &b) < 0) goto fail;
	}
	if (!(c->tj = tj3Init(TJINIT_DECOMPRESS)) || tb_init(&c->tb, (size_t)w * h * 3)) goto fail;
	enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (xioctl(c->fd, VIDIOC_STREAMON, &t) < 0) {
		fprintf(stderr, "webcam: VIDIOC_STREAMON: %s%s\n", strerror(errno),
			errno == ENOSPC ? " (USB bus out of isochronous bandwidth: use a port on another controller)" : "");
		goto fail;
	}
	if (pthread_create(&c->thr, NULL, cam_thread, c)) goto fail;
	return c;
fail:
	close(c->fd);
	free(c);
	return NULL;
}

const uint8_t *webcam_acquire(webcam *c, double *t, uint32_t *seq)
{
	int s = tb_acquire_slot(&c->tb);
	if (s < 0) return NULL;
	if (t) *t = c->slot_t[s];
	if (seq) *seq = c->slot_seq[s];
	return c->tb.buf[s];
}

void webcam_stats(webcam *c, unsigned *frames, unsigned *fails, unsigned *gaps, double *decode_ms_mean)
{
	unsigned n = atomic_load(&c->frames);
	*frames = n;
	*fails = atomic_load(&c->fails);
	*gaps = atomic_load(&c->gaps);
	*decode_ms_mean = n ? (double)atomic_load(&c->decode_ns) * 1e-6 / n : 0.0;
}

const char *webcam_path(const webcam *c) { return c->path; }

void webcam_close(webcam *c)
{
	if (!c) return;
	enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	atomic_store(&c->die, 1);
	pthread_join(c->thr, NULL);
	xioctl(c->fd, VIDIOC_STREAMOFF, &t);
	for (unsigned i = 0; i < c->nbuf; i++) munmap(c->map[i], c->len[i]);
	close(c->fd);
	tj3Destroy(c->tj);
	free(c);
}

unsigned webcam_warnings(webcam *c) { return atomic_load(&c->warnings); }
const char *webcam_error(webcam *c) { return atomic_load(&c->have_err) ? c->err : ""; }
