/* flow.c: see flow.h. Same algorithm and parameters as OpenCV calcOpticalFlowPyrLK (pyrDown 5x5
 * reflect-101 pyramid, Scharr gradients, 11x11 window, 4 levels, <= 10 iterations, 0.01 px stop),
 * evaluated for every pixel, both directions, then a forward-backward consistency gate. */
#define _GNU_SOURCE
#include "flow.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LV = 4, NQ = 4 };

#define CS_HEAD "#version 460 core\nlayout(local_size_x = 16, local_size_y = 16) in;\n"
#define REFL "int refl(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }\n"

static const char SRC_GRAY[] = CS_HEAD
	"layout(binding = 8) uniform sampler2D s_rgb;\n"
	"layout(binding = 2, r32f) uniform writeonly image2D i_out;\n"
	"uniform ivec2 u_size;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (p.x >= u_size.x || p.y >= u_size.y) return;\n"
	"  vec3 c = texelFetch(s_rgb, p, 0).rgb * 255.0;\n"
	"  imageStore(i_out, p, vec4(round(dot(c, vec3(0.299, 0.587, 0.114))), 0.0, 0.0, 0.0));\n"
	"}\n";

static const char SRC_PYR[] = CS_HEAD REFL
	"layout(binding = 8) uniform sampler2D s_src;\n"
	"layout(binding = 2, r32f) uniform writeonly image2D i_out;\n"
	"uniform ivec2 u_src;\n"
	"uniform ivec2 u_size;\n"
	"const float K[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (p.x >= u_size.x || p.y >= u_size.y) return;\n"
	"  float s = 0.0;\n"
	"  for (int j = 0; j < 5; j++) {\n"
	"    int y = refl(2 * p.y + j - 2, u_src.y);\n"
	"    float r = 0.0;\n"
	"    for (int i = 0; i < 5; i++) r += K[i] * texelFetch(s_src, ivec2(refl(2 * p.x + i - 2, u_src.x), y), 0).r;\n"
	"    s += K[j] * r;\n"
	"  }\n"
	"  imageStore(i_out, p, vec4(s * (1.0 / 256.0), 0.0, 0.0, 0.0));\n"
	"}\n";

static const char SRC_SCHARR[] = CS_HEAD REFL
	"layout(binding = 8) uniform sampler2D s_src;\n"
	"layout(binding = 2, rg32f) uniform writeonly image2D i_out;\n"
	"uniform ivec2 u_size;\n"
	"float at(int x, int y) { return texelFetch(s_src, ivec2(refl(x, u_size.x), refl(y, u_size.y)), 0).r; }\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (p.x >= u_size.x || p.y >= u_size.y) return;\n"
	"  int x = p.x, y = p.y;\n"
	"  float gx = 3.0 * (at(x + 1, y - 1) - at(x - 1, y - 1)) + 10.0 * (at(x + 1, y) - at(x - 1, y)) + 3.0 * (at(x + 1, y + 1) - at(x - 1, y + 1));\n"
	"  float gy = 3.0 * (at(x - 1, y + 1) - at(x - 1, y - 1)) + 10.0 * (at(x, y + 1) - at(x, y - 1)) + 3.0 * (at(x + 1, y + 1) - at(x + 1, y - 1));\n"
	"  imageStore(i_out, p, vec4(gx, gy, 0.0, 0.0) * (1.0 / 32.0));\n"
	"}\n";

/* inverse-compositional translation LK: F <- F - H^-1 sum g (J(x+F) - T(x)) */
static const char SRC_LK[] = CS_HEAD
	"layout(binding = 8) uniform sampler2D s_tmpl;\n"
	"layout(binding = 9) uniform sampler2D s_grad;\n"
	"layout(binding = 10) uniform sampler2D s_tgt;\n"
	"layout(binding = 11) uniform sampler2D s_init;\n"
	"layout(binding = 2, rgba32f) uniform writeonly image2D i_out;\n"
	"uniform ivec2 u_size;\n"
	"uniform int u_init;\n"
	"const int R = 5;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (p.x >= u_size.x || p.y >= u_size.y) return;\n"
	"  vec2 inv = 1.0 / vec2(u_size);\n"
	"  vec2 F = u_init != 0 ? 2.0 * texture(s_init, (vec2(p) + 0.5) * inv).xy : vec2(0.0);\n"
	"  float a = 0.0, b = 0.0, c = 0.0;\n"
	"  for (int dy = -R; dy <= R; dy++)\n"
	"    for (int dx = -R; dx <= R; dx++) {\n"
	"      vec2 g = texelFetch(s_grad, clamp(p + ivec2(dx, dy), ivec2(0), u_size - 1), 0).xy;\n"
	"      a += g.x * g.x; b += g.x * g.y; c += g.y * g.y;\n"
	"    }\n"
	"  float n = float((2 * R + 1) * (2 * R + 1));\n"
	"  float mineig = 0.5 * (a + c - sqrt((a - c) * (a - c) + 4.0 * b * b)) / n;\n"
	"  float det = a * c - b * b;\n"
	"  bool ok = det > 1e-3 * n * n;\n"
	"  if (ok) {\n"
	"    float ia = c / det, ib = -b / det, ic = a / det;\n"
	"    for (int it = 0; it < 10; it++) {\n"
	"      vec2 e = vec2(0.0);\n"
	"      for (int dy = -R; dy <= R; dy++)\n"
	"        for (int dx = -R; dx <= R; dx++) {\n"
	"          ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), u_size - 1);\n"
	"          float j = texture(s_tgt, (vec2(q) + F + 0.5) * inv).r;\n"
	"          e += texelFetch(s_grad, q, 0).xy * (j - texelFetch(s_tmpl, q, 0).r);\n"
	"        }\n"
	"      vec2 d = vec2(ia * e.x + ib * e.y, ib * e.x + ic * e.y);\n"
	"      F -= d;\n"
	"      if (dot(d, d) < 1e-4) break;\n"
	"    }\n"
	"  }\n"
	"  imageStore(i_out, p, vec4(F, mineig, ok ? 1.0 : 0.0));\n"
	"}\n";

static const char SRC_FB[] = CS_HEAD
	"layout(binding = 8) uniform sampler2D s_bwd;\n"
	"layout(binding = 9) uniform sampler2D s_fwd;\n"
	"layout(binding = 2, rgba32f) uniform writeonly image2D i_out;\n"
	"uniform ivec2 u_size;\n"
	"uniform float u_fb;\n"
	"uniform float u_mineig;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (p.x >= u_size.x || p.y >= u_size.y) return;\n"
	"  vec4 b = texelFetch(s_bwd, p, 0);\n"
	"  vec2 q = vec2(p) + 0.5 + b.xy;\n"
	"  bool inb = all(greaterThanEqual(q, vec2(0.0))) && all(lessThan(q, vec2(u_size)));\n"
	"  float err = length(b.xy + texture(s_fwd, q / vec2(u_size)).xy);\n"
	"  bool ok = b.w > 0.5 && inb && err < u_fb && b.z >= u_mineig;\n"
	"  imageStore(i_out, p, vec4(b.xy, err, ok ? 1.0 : 0.0));\n"
	"}\n";

static const struct { const char *file; const char *src; } FL_STAGES[] = {
	{"fl_gray.comp", SRC_GRAY}, {"fl_pyr.comp", SRC_PYR}, {"fl_scharr.comp", SRC_SCHARR},
	{"fl_lk.comp", SRC_LK}, {"fl_fb.comp", SRC_FB},
};
enum { P_GRAY, P_PYR, P_SCHARR, P_LK, P_FB, NP };

static GLuint progs[NP];
static float g_fb = 0.5f, g_mineig = 4.0f;

struct flowcam {
	int w[LV], h[LV], newest, pushed;
	GLuint gray[2][LV], grad[2][LV], lk[2][LV], out;
	GLuint q[NQ];
	int qi, qbusy[NQ];
	double t[2], gpu_ms_sum;
	unsigned gpu_n;
	char name[16];
};

int flow_dump_shaders(const char *dir)
{
	for (int i = 0; i < NP; i++) {
		char path[1024];
		snprintf(path, sizeof path, "%s/%s", dir, FL_STAGES[i].file);
		FILE *f = fopen(path, "w");
		if (!f || fputs(FL_STAGES[i].src, f) < 0 || fclose(f)) { perror(path); return 1; }
	}
	return 0;
}

void flow_set(float fb_max_px, float min_eig) { g_fb = fb_max_px; g_mineig = min_eig; }

static GLuint build(int i)
{
	GLuint s = glCreateShader(GL_COMPUTE_SHADER), p = glCreateProgram();
	const char *src = FL_STAGES[i].src;
	GLint ok = 0, len = 0;
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
	if (!ok || len > 1) {
		char log[4096];
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		fprintf(stderr, "flow: shader %s %s:\n%s\n", FL_STAGES[i].file, ok ? "has diagnostics" : "failed", log);
		exit(2);
	}
	glAttachShader(p, s);
	glLinkProgram(p);
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
	if (!ok || len > 1) {
		char log[4096];
		glGetProgramInfoLog(p, sizeof log, NULL, log);
		fprintf(stderr, "flow: link %s:\n%s\n", FL_STAGES[i].file, log);
		exit(2);
	}
	return p;
}

static GLuint tex(GLenum fmt, int w, int h)
{
	GLuint t;
	glCreateTextures(GL_TEXTURE_2D, 1, &t);
	glTextureStorage2D(t, 1, fmt, w, h);
	glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(t, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return t;
}

flowcam *flow_create(int w, int h, const char *name)
{
	flowcam *f = calloc(1, sizeof *f);
	if (!f) return NULL;
	if (!progs[0])
		for (int i = 0; i < NP; i++) progs[i] = build(i);
	snprintf(f->name, sizeof f->name, "%s", name);
	for (int l = 0; l < LV; l++) {
		f->w[l] = l ? (f->w[l - 1] + 1) / 2 : w;
		f->h[l] = l ? (f->h[l - 1] + 1) / 2 : h;
		for (int s = 0; s < 2; s++) {
			f->gray[s][l] = tex(GL_R32F, f->w[l], f->h[l]);
			f->grad[s][l] = tex(GL_RG32F, f->w[l], f->h[l]);
			f->lk[s][l] = tex(GL_RGBA32F, f->w[l], f->h[l]);
		}
	}
	f->out = tex(GL_RGBA32F, w, h);
	glCreateQueries(GL_TIME_ELAPSED, NQ, f->q);
	return f;
}

static void run(int prog, int w, int h)
{
	glUseProgram(progs[prog]);
	glProgramUniform2i(progs[prog], glGetUniformLocation(progs[prog], "u_size"), w, h);
	glDispatchCompute((GLuint)(w + 15) / 16, (GLuint)(h + 15) / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

static void lk(flowcam *f, int dir, int tmpl, int tgt)
{
	for (int l = LV - 1; l >= 0; l--) {
		glBindTextureUnit(8, f->gray[tmpl][l]);
		glBindTextureUnit(9, f->grad[tmpl][l]);
		glBindTextureUnit(10, f->gray[tgt][l]);
		glBindTextureUnit(11, l + 1 < LV ? f->lk[dir][l + 1] : f->out); /* never the image being written */
		glBindImageTexture(2, f->lk[dir][l], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
		glProgramUniform1i(progs[P_LK], glGetUniformLocation(progs[P_LK], "u_init"), l + 1 < LV);
		run(P_LK, f->w[l], f->h[l]);
	}
}

/* pyramid levels 1.. and Scharr gradients of slot s, from its level-0 gray */
static void build_pyr(flowcam *f, int s)
{
	for (int l = 1; l < LV; l++) {
		glBindTextureUnit(8, f->gray[s][l - 1]);
		glBindImageTexture(2, f->gray[s][l], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
		glProgramUniform2i(progs[P_PYR], glGetUniformLocation(progs[P_PYR], "u_src"), f->w[l - 1], f->h[l - 1]);
		run(P_PYR, f->w[l], f->h[l]);
	}
	for (int l = 0; l < LV; l++) {
		glBindTextureUnit(8, f->gray[s][l]);
		glBindImageTexture(2, f->grad[s][l], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
		run(P_SCHARR, f->w[l], f->h[l]);
	}
}

/* both directions (template slot a: a -> b, and b -> a) and the forward-backward gate into out */
static void solve_pair(flowcam *f, int a, int b)
{
	lk(f, 0, a, b);
	lk(f, 1, b, a);
	glBindTextureUnit(8, f->lk[0][0]);
	glBindTextureUnit(9, f->lk[1][0]);
	glBindImageTexture(2, f->out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glProgramUniform1f(progs[P_FB], glGetUniformLocation(progs[P_FB], "u_fb"), g_fb);
	glProgramUniform1f(progs[P_FB], glGetUniformLocation(progs[P_FB], "u_mineig"), g_mineig);
	run(P_FB, f->w[0], f->h[0]);
}

static void timer_begin(flowcam *f)
{
	if (f->qbusy[f->qi]) {
		GLuint64 ns = 0;
		glGetQueryObjectui64v(f->q[f->qi], GL_QUERY_RESULT, &ns);
		f->gpu_ms_sum += (double)ns * 1e-6;
		f->gpu_n++;
	}
	glBeginQuery(GL_TIME_ELAPSED, f->q[f->qi]);
}

static void timer_end(flowcam *f)
{
	glEndQuery(GL_TIME_ELAPSED);
	f->qbusy[f->qi] = 1;
	f->qi = (f->qi + 1) % NQ;
}

void flow_push(flowcam *f, GLuint rgb_tex, double t)
{
	int s = f->pushed ? f->newest ^ 1 : 0;
	timer_begin(f);
	glBindTextureUnit(8, rgb_tex);
	glBindImageTexture(2, f->gray[s][0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
	run(P_GRAY, f->w[0], f->h[0]);
	build_pyr(f, s);
	if (f->pushed) solve_pair(f, s, s ^ 1); /* backward newest -> previous, forward, gate */
	timer_end(f);
	f->t[s] = t;
	f->newest = s;
	f->pushed++;
}

void flow_pair(flowcam *f, GLuint gray_a, GLuint gray_b)
{
	timer_begin(f);
	glCopyImageSubData(gray_a, GL_TEXTURE_2D, 0, 0, 0, 0, f->gray[0][0], GL_TEXTURE_2D, 0, 0, 0, 0, f->w[0], f->h[0], 1);
	glCopyImageSubData(gray_b, GL_TEXTURE_2D, 0, 0, 0, 0, f->gray[1][0], GL_TEXTURE_2D, 0, 0, 0, 0, f->w[0], f->h[0], 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
	build_pyr(f, 0);
	build_pyr(f, 1);
	solve_pair(f, 0, 1);
	timer_end(f);
	f->newest = 0;
	f->pushed = 2;
}
int flow_ready(const flowcam *f) { return f->pushed >= 2; }
GLuint flow_result(const flowcam *f) { return f->out; }
double flow_dt(const flowcam *f) { return f->t[f->newest] - f->t[f->newest ^ 1]; }
GLuint flow_gray(const flowcam *f, int newest) { return f->gray[newest ? f->newest : f->newest ^ 1][0]; }
double flow_gpu_ms(const flowcam *f) { return f->gpu_n ? f->gpu_ms_sum / f->gpu_n : 0.0; }
