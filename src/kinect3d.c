/* kinect3d.c: live textured 3D view of an Xbox 360 Kinect (model 1414).
 *
 * Geometry: FREENECT_DEPTH_MM in the depth camera's own frame, back-projected with the device's
 * factory zero-plane calibration, the same formula as freenect_camera_to_world:
 *     X = (x - 320) * k * z,  Y = (y - 240) * k * z,  k = 2 * reference_pixel_size / reference_distance
 * Colour: libfreenect's registration (registration_table + depth_to_rgb_shift, the integer arithmetic
 * of freenect_map_rgb_to_depth) evaluated per vertex on the GPU, with the same RGB-space z-buffer
 * occlusion test, built per depth frame by a compute shader with imageAtomicMin.
 * Mesh: the 640x480 depth grid as 639x479 quads = 612,162 triangles; a triangle is dropped when a
 * corner has no depth or its depth spread exceeds cull * nearest depth (silhouette edges).
 *
 * Two mesh paths, chosen at run time (M), compared by --selftest and timed by --bench:
 *   A  index buffer + geometry shader that drops bad triangles (the plain form)
 *   B  no buffers at all: gl_VertexID -> quad and corner through two 6-bit tables (0x1A, 0x34);
 *      MEASURED SLOWER (median 0.079-0.081 vs 0.049 ms, RTX 3090 Ti): kept only as the selftest oracle;
 *      every vertex re-evaluates its whole triangle, so dropping is exact without a geometry shader
 *
 * Capture: libfreenect event thread; zero-copy triple buffers (one atomic word: middle index + fresh
 * bit) swap the buffer libfreenect fills, so frames are never copied on the CPU.
 *
 * Usage: kinect3d [--method a|b] [--selftest] [--bench FRAMES] [--dump-shaders DIR]
 * Mouse: left drag orbit, right drag pan, wheel zoom.  Keys: R reset to the Kinect's own viewpoint,
 * M mesh path, O occlusion grey on/off, [ ] edge cull threshold, W/X tilt motor up/down, S level,
 * Esc or Q quit.
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>
#include <libfreenect.h>
#include <libfreenect_registration.h>
#include "flow.h"
#include "tribuf.h"
#include "webcam.h"
#include "facetrack.h"
#include "fsrflow.h"
#include "vtface.h"

enum { W = 640, H = 480, NPIX = W * H, NQUAD = (W - 1) * (H - 1), NVERT = NQUAD * 6, ZMAX = 10000 };
#define DEFAULT_METHOD 0 /* A: measured 1.60-1.65x faster than B (README, out/bench-*.log) */

/* ==== GLSL ====================================================================================== */
static const char GLSL_COMMON[] =
	"#version 460 core\n"
	"layout(binding = 0) uniform usampler2D t_depth;\n" /* R16UI depth in mm */
	"layout(binding = 1) uniform isampler2D t_reg;\n"   /* RG32I registration_table, y pre-offset */
	"layout(binding = 2) uniform isampler2D t_shift;\n" /* R32I depth_to_rgb_shift[10000] */
	"layout(binding = 3) uniform usampler2D t_zmin;\n"  /* R32UI nearest depth per RGB pixel */
	"layout(binding = 4) uniform sampler2D t_rgb;\n"
	"uniform mat4 u_mvp;\n"
	/* geometry at the colour's instant: a vertex moves by its velocity (sflow, room frame, mm/s) times u_tshift =
	 * instant of the Kinect colour on screen - instant of the depth frame (s); only above the measured noise floor */
	"layout(binding = 16) uniform sampler2D t_velm;\n"
	"uniform float u_tshift;\n"
	"uniform mat3 u_Rcam;\n"
	"uniform float u_k;\n"
	"uniform float u_cull;\n"
	"uniform int u_occl;\n"
	"struct Pix { uint wz; bool ok; int s; int cx; int cy; bool inb; };\n"
	"Pix pix(ivec2 p) {\n"
	"  Pix r;\n"
	"  r.wz = texelFetch(t_depth, p, 0).r;\n"
	"  r.ok = r.wz != 0u && r.wz < 10000u;\n"
	"  ivec2 rg = texelFetch(t_reg, p, 0).xy;\n"
	"  r.s = rg.x + texelFetch(t_shift, ivec2(int(min(r.wz, 9999u)), 0), 0).r;\n"
	"  r.cx = r.s >= 0 ? (r.s >> 8) : -((-r.s) >> 8);\n" /* C division by 256, toward zero */
	"  r.cy = rg.y;\n"
	"  r.inb = r.cx >= 0 && r.cx < 640 && r.cy >= 0 && r.cy < 480;\n"
	"  return r;\n"
	"}\n"
	"vec4 clip_of(ivec2 p, uint wz) {\n"
	"  float z = float(wz);\n"
	"  vec3 X = vec3(float(p.x - 320) * u_k * z, float(p.y - 240) * u_k * z, z);\n"
	"  if (u_tshift != 0.0) {\n"
	"    vec4 v = texelFetch(t_velm, p, 0);\n"
	"    float zm = z * 1e-3;\n"
	"    if (v.w > 0.5 && length(v.xyz) >= 150.0 + 40.0 * zm * zm) X += u_tshift * (transpose(u_Rcam) * v.xyz);\n"
	"  }\n"
	"  return u_mvp * vec4(X.x, -X.y, -X.z, 1.0);\n"
	"}\n"
	"bool keep3(uint a, uint b, uint c) {\n"
	"  uint mn = min(a, min(b, c)), mx = max(a, max(b, c));\n"
	"  return mn != 0u && mx < 10000u && float(mx - mn) <= u_cull * float(mn);\n"
	"}\n"
	"void attrs(Pix q, out vec2 uv, out float occl) {\n"
	"  uv = vec2(float(q.s) * (1.0 / (256.0 * 640.0)), (float(q.cy) + 0.5) * (1.0 / 480.0));\n"
	"  occl = 1.0;\n"
	"  if (q.ok && q.inb)\n"
	/* occluded = behind the nearest surface at this RGB pixel by more than max(50 mm, 3 depth steps): two
	 * depth pixels one quantisation step apart often share an RGB pixel; the exact z-buffer test (as in
	 * freenect_map_rgb_to_depth) then marks the farther one and paints contour lines over every surface */
	"    occl = (u_occl != 0 && float(q.wz) > float(texelFetch(t_zmin, ivec2(q.cx, q.cy), 0).r)\n"
	"                                 + max(50.0, 3.0 * float(q.wz) * float(q.wz) / 348000.0)) ? 1.0 : 0.0;\n"
	"}\n"
	/* webcam: calibration from tools/kcalib.cpp (OpenCV model, pixel centres at integers) */
	"layout(binding = 6) uniform usampler2D t_win;\n"  /* R32UI depth-pixel index that won each RGB pixel */
	"layout(binding = 7) uniform usampler2D t_wz;\n"   /* R32UI 320x240 nearest depth seen by the webcam */
	"layout(binding = 12) uniform sampler2D t_wrgb;\n"
	"uniform int u_wcal;\n"
	"uniform mat3 u_wR;\n"
	"uniform vec3 u_wt;\n"
	"uniform vec4 u_wK;\n"   /* fx fy cx cy */
	"uniform vec2 u_wd;\n"   /* k1 k2 */
	"uniform vec3 u_wgain;\n"
	"uniform vec3 u_woff;\n"
	"vec3 kworld(ivec2 p, uint wz) { float z = float(wz); return vec3(float(p.x - 320) * u_k * z, float(p.y - 240) * u_k * z, z); }\n"
	"vec3 wproj(vec3 X) {\n" /* webcam pixel (x, y) and depth in the webcam frame (z, mm) */
	"  vec3 c = u_wR * X + u_wt;\n"
	"  vec2 n = c.xy / c.z;\n"
	"  float r2 = dot(n, n), d = 1.0 + r2 * (u_wd.x + r2 * u_wd.y);\n"
	"  return vec3(u_wK.x * n.x * d + u_wK.z, u_wK.y * n.y * d + u_wK.w, c.z);\n"
	"}\n"
	"ivec2 whalf(vec2 w) { return ivec2(floor((w + 0.5) * 0.5)); }\n"
	"bool whalf_in(ivec2 h) { return all(greaterThanEqual(h, ivec2(0))) && all(lessThan(h, ivec2(320, 240))); }\n"
	"void wattrs(ivec2 p, Pix q, out vec2 wuv, out float wvis) {\n"
	"  wuv = vec2(0.0);\n"
	"  wvis = 0.0;\n"
	"  if (u_wcal == 0 || !q.ok) return;\n"
	"  vec3 w = wproj(kworld(p, q.wz));\n"
	"  ivec2 h = whalf(w.xy);\n"
	"  if (w.z <= 50.0 || !whalf_in(h)) return;\n"
	"  wuv = (w.xy + 0.5) / vec2(640.0, 480.0);\n"
	"  wvis = w.z <= float(texelFetch(t_wz, h, 0).r) * 1.02 + 15.0 ? 1.0 : 0.0;\n"
	"}\n";

static const char BODY_MESH_A_VERT[] =
	"invariant gl_Position;\n"
	"out vec2 v_uv;\n"
	"out float v_occl;\n"
	"out vec2 v_wuv;\n"
	"out vec3 v_X;\n"
	"out float v_wvis;\n"
	"flat out uint v_wz;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_VertexID % 640, gl_VertexID / 640);\n"
	"  Pix me = pix(p);\n"
	"  attrs(me, v_uv, v_occl);\n"
	"  wattrs(p, me, v_wuv, v_wvis);\n"
	"  v_X = kworld(p, me.wz);\n"
	"  v_wz = me.wz;\n"
	"  gl_Position = clip_of(p, me.wz);\n"
	"}\n";

static const char BODY_MESH_A_GEOM[] =
	"layout(triangles) in;\n"
	"layout(triangle_strip, max_vertices = 3) out;\n"
	"in vec2 v_uv[];\n"
	"in float v_occl[];\n"
	"in vec2 v_wuv[];\n"
	"in vec3 v_X[];\n"
	"out vec3 f_X;\n"
	"in float v_wvis[];\n"
	"flat in uint v_wz[];\n"
	"out vec2 f_uv;\n"
	"out float f_occl;\n"
	"out vec2 f_wuv;\n"
	"out float f_wvis;\n"
	"void main() {\n"
	"  if (!keep3(v_wz[0], v_wz[1], v_wz[2])) return;\n"
	"  for (int i = 0; i < 3; i++) {\n"
	"    gl_Position = gl_in[i].gl_Position;\n"
	"    f_uv = v_uv[i];\n"
	"    f_occl = v_occl[i];\n"
	"    f_wuv = v_wuv[i];\n"
	"    f_wvis = v_wvis[i];\n"
	"    f_X = v_X[i];\n"
	"    EmitVertex();\n"
	"  }\n"
	"  EndPrimitive();\n"
	"}\n";

/* corner c of a quad: (0,0) (1,0) (0,1) | (1,0) (1,1) (0,1); dx bits 0x1A = 011010b, dy bits 0x34 = 110100b */
static const char BODY_MESH_B_VERT[] =
	"invariant gl_Position;\n"
	"out vec2 f_uv;\n"
	"out float f_occl;\n"
	"out vec2 f_wuv;\n"
	"out float f_wvis;\n"
	"ivec2 corner(int c) { return ivec2((0x1A >> c) & 1, (0x34 >> c) & 1); }\n"
	"void main() {\n"
	"  int q = gl_VertexID / 6, c = gl_VertexID - q * 6, t0 = c >= 3 ? 3 : 0;\n"
	"  ivec2 base = ivec2(q % 639, q / 639);\n"
	"  ivec2 p = base + corner(c);\n"
	"  Pix me = pix(p);\n"
	"  bool k = keep3(texelFetch(t_depth, base + corner(t0), 0).r,\n"
	"                 texelFetch(t_depth, base + corner(t0 + 1), 0).r,\n"
	"                 texelFetch(t_depth, base + corner(t0 + 2), 0).r);\n"
	"  attrs(me, f_uv, f_occl);\n"
	"  wattrs(p, me, f_wuv, f_wvis);\n"
	"  gl_Position = k ? clip_of(p, me.wz) : vec4(2.0, 2.0, 2.0, 1.0);\n"
	"}\n";

/* u_occl: 0 Kinect colour everywhere, 1 grey where the Kinect RGB cannot see, 2 webcam fills those
 * places where the webcam sees them (grey where neither does), 3 webcam colour wherever it sees
 * (calibration check: misalignment shows at every edge) */
static const char BODY_MESH_FRAG[] =
	"in vec2 f_uv;\n"
	"in float f_occl;\n"
	"in vec2 f_wuv;\n"
	"in float f_wvis;\n"
	"out vec4 o;\n"
	"void main() {\n"
	"  vec3 grey = vec3(0.30, 0.30, 0.33);\n"
	"  vec3 kc = texture(t_rgb, f_uv).rgb;\n"
	"  vec3 wc = clamp(texture(t_wrgb, f_wuv).rgb * u_wgain + u_woff, 0.0, 1.0);\n"
	"  float ww = u_occl >= 2 ? f_wvis : 0.0;\n"
	"  vec3 c = u_occl == 3 ? mix(mix(kc, grey, f_occl), wc, ww)\n"
	"                       : kc * (1.0 - f_occl) + wc * (f_occl * ww) + grey * (f_occl * (1.0 - ww));\n"
	"  o = vec4(c, 1.0);\n"
	"}\n";

/* depth filter (before every other pass): the Kinect marks invalid depth in 8-pixel blocks (measured:
 * raw 11-bit and mm alike, ~380 runs of exactly 8 per frame). Gap fill: an invalid pixel with valid
 * depth on both sides within u_fill px (rows first, then columns) is interpolated when the two sides
 * agree within max(30 mm, 3 %), so silhouettes are never bridged. Temporal: a new depth within 3
 * quantisation steps (z^2 / 348000 mm) of the history is blended (u_alpha); larger changes pass at
 * once (motion). A pixel that drops out keeps its last depth for up to 2 frames. State: rg32f (z, age). */
static const char BODY_DFILL_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 13) uniform usampler2D t_draw;\n"
	"layout(binding = 6, r16ui) uniform writeonly uimage2D i_dout;\n"
	"layout(binding = 7, rg32f) uniform image2D i_dtemp;\n"
	"uniform int u_fill;\n"
	"uniform float u_alpha;\n"
	"uint dr(ivec2 p) { return texelFetch(t_draw, clamp(p, ivec2(0), ivec2(639, 479)), 0).r; }\n"
	"bool dok(uint z) { return z != 0u && z < 10000u; }\n"
	"float axis(ivec2 p, ivec2 dir, out bool ok) {\n" /* plane-exact estimate across a hole along dir */
	"  int a = 0, b = 0; uint za = 0u, zb = 0u;\n"
	"  ok = false;\n"
	"  for (int i = 1; i <= 16; i++) {\n"
	"    if (i > u_fill) break;\n"
	"    if (a == 0) { uint z = dr(p - dir * i); if (dok(z)) { a = i; za = z; } }\n"
	"    if (b == 0) { uint z = dr(p + dir * i); if (dok(z)) { b = i; zb = z; } }\n"
	"  }\n"
	"  if (a == 0 || b == 0) return 0.0;\n"
	"  float fa = float(za), fb = float(zb);\n"
	"  if (abs(fa - fb) > max(30.0, 0.03 * min(fa, fb))) return 0.0;\n"
	"  ok = true;\n"
	"  return mix(fa, fb, float(a) / float(a + b));\n"
	"}\n"
	"float fill2d(ivec2 p) {\n" /* at least 2 of the 4 axes must agree: inside a surface, never at a silhouette */
	"  const ivec2 D[4] = ivec2[4](ivec2(1, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, -1));\n"
	"  float s = 0.0;\n"
	"  int n = 0;\n"
	"  for (int k = 0; k < 4; k++) { bool ok; float e = axis(p, D[k], ok); if (ok) { s += e; n++; } }\n"
	"  return n >= 2 ? s / float(n) : 0.0;\n"
	"}\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  uint z = dr(p);\n"
	"  float zf = dok(z) ? float(z) : 0.0;\n"
	"  if (zf == 0.0 && u_fill > 0) {\n"
	"    zf = fill2d(p);\n"
	"  }\n"
	"  vec2 h = imageLoad(i_dtemp, p).rg;\n"
	"  float age = 0.0;\n"
	"  if (zf > 0.0 && h.r > 0.0 && abs(zf - h.r) <= 3.0 * zf * zf / 348000.0) zf = mix(h.r, zf, u_alpha);\n"
	"  else if (zf == 0.0 && h.r > 0.0 && h.g < 2.0 && u_alpha < 1.0) { zf = h.r; age = h.g + 1.0; }\n"
	"  imageStore(i_dtemp, p, vec4(zf, age, 0.0, 0.0));\n"
	"  imageStore(i_dout, p, uvec4(uint(round(zf)), 0u, 0u, 0u));\n"
	"}\n";

/* inverse registration: which depth pixel won each RGB pixel's z-buffer (lowest index on ties) */
static const char BODY_WINNER_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 3, r32ui) uniform uimage2D i_win;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  if (q.ok && q.inb && q.wz <= texelFetch(t_zmin, ivec2(q.cx, q.cy), 0).r)\n"
	"    imageAtomicMin(i_win, ivec2(q.cx, q.cy), uint(p.y * 640 + p.x));\n"
	"}\n";

/* calibration snapshot: the 3D point (Kinect frame, mm) seen at each Kinect RGB pixel, w = 1 */
static const char BODY_PMAP_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 4, rgba32f) uniform writeonly image2D i_pmap;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  if (q.ok && q.inb && texelFetch(t_win, ivec2(q.cx, q.cy), 0).r == uint(p.y * 640 + p.x))\n"
	"    imageStore(i_pmap, ivec2(q.cx, q.cy), vec4(kworld(p, q.wz), 1.0));\n"
	"}\n";

/* webcam z-buffer at half resolution: nearest depth (webcam frame, mm) per 2x2 webcam pixels */
static const char BODY_WZ_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 5, r32ui) uniform uimage2D i_wz;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  if (!q.ok) return;\n"
	"  vec3 w = wproj(kworld(p, q.wz));\n"
	"  ivec2 h = whalf(w.xy);\n"
	"  if (w.z > 50.0 && whalf_in(h)) imageAtomicMin(i_wz, h, uint(w.z));\n"
	"}\n";

static const char BODY_SCATTER_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 0, r32ui) uniform uimage2D i_zmin;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  if (q.ok && q.inb) imageAtomicMin(i_zmin, ivec2(q.cx, q.cy), q.wz);\n"
	"}\n";

/* selftest only: per depth pixel the colour freenect_map_rgb_to_depth would produce
 * (black = no depth or out of bounds, magenta = occluded, else the RGB texel) */
static const char BODY_CHECK_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 1, rgba8) uniform writeonly image2D i_out;\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  vec4 c = vec4(0.0, 0.0, 0.0, 1.0);\n"
	"  if (q.ok && q.inb)\n"
	"    c = q.wz <= texelFetch(t_zmin, ivec2(q.cx, q.cy), 0).r\n"
	"      ? vec4(texelFetch(t_rgb, ivec2(q.cx, q.cy), 0).rgb, 1.0) : vec4(1.0, 0.0, 1.0, 1.0);\n"
	"  imageStore(i_out, p, c);\n"
	"}\n";

/* picture-in-picture panels: camera feeds (mode 0) or flow as hue = direction, value = speed (mode 1) */
static const char BODY_PIP_VERT[] =
	"uniform vec4 u_rect;\n"
	"out vec2 v_uv;\n"
	"void main() {\n"
	"  vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);\n"
	"  v_uv = vec2(c.x, 1.0 - c.y);\n"
	"  gl_Position = vec4(mix(u_rect.xy, u_rect.zw, c), 0.0, 1.0);\n"
	"}\n";

static const char BODY_PIP_FRAG[] =
	"layout(binding = 5) uniform sampler2D t_pip;\n"
	"uniform int u_mode;\n"
	"in vec2 v_uv;\n"
	"out vec4 o;\n"
	"vec3 hsv(float h, float s, float v) {\n"
	"  vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);\n"
	"  return v * mix(vec3(1.0), k, s);\n"
	"}\n"
	/* the face tracker (VisionTrack) on a camera feed: its box, member features green, the rest on probation grey */
	"uniform vec4 u_box;\n"   /* x0 y0 x1 y1, pixels of the camera; x1 <= x0: none */
	"uniform vec3 u_boxcol;\n"
	"uniform vec2 u_feat[64];\n"
	"uniform int u_nfeat, u_nmem;\n"
	"void main() {\n"
	"  vec4 t = texture(t_pip, v_uv);\n"
	"  if (u_mode == 0) {\n"
	"    o = vec4(t.rgb, 1.0);\n"
	"    vec2 p = v_uv * vec2(textureSize(t_pip, 0));\n"
	"    for (int i = 0; i < u_nfeat; i++)\n"
	"      if (dot(p - u_feat[i], p - u_feat[i]) < 6.25) o = vec4(i < u_nmem ? vec3(0.2, 1.0, 0.2) : vec3(0.6), 1.0);\n"
	"    if (u_box.z > u_box.x) {\n"
	"      vec2 e = vec2(2.5);\n"
	"      bool outer = all(greaterThan(p, u_box.xy - e)) && all(lessThan(p, u_box.zw + e));\n"
	"      bool inner = all(greaterThan(p, u_box.xy + e)) && all(lessThan(p, u_box.zw - e));\n"
	"      if (outer && !inner) o = vec4(u_boxcol, 1.0);\n"
	"    }\n"
	"    return;\n"
	"  }\n"
	"  o = t.w > 0.5 ? vec4(hsv(atan(t.y, t.x) * 0.15915494 + 0.5, 1.0, clamp(length(t.xy) * 0.125, 0.0, 1.0)), 1.0)\n"
	"               : vec4(0.15, 0.15, 0.15, 1.0);\n"
	"}\n";

/* ==== flow lock: keep the webcam pose locked to the Kinect model, every webcam frame ==============
 * syn.frag: the Kinect mesh drawn from the webcam's pose: gray in the webcam's colour space (inverse
 * of the calibrated gain/offset), -1 where the Kinect RGB cannot see; and the 3D point (Kinect frame).
 * comp.comp: holes take the real webcam gray (zero residual there, and they carry no 3D point).
 * gn.comp: flow synthetic -> real gives, per pixel, observed - predicted webcam position r; with the
 * 2x6 Jacobian of the projection under a pose increment (rotation w, translation dt: c' = exp(w) c + dt)
 * each workgroup sums Huber-weighted J^T J (21), J^T r (6), sum |r|, count. */
static const char BODY_SYN_FRAG[] =
	"in vec2 f_uv;\n"
	"in float f_occl;\n"
	"in vec3 f_X;\n"
	"layout(location = 0) out float o_gray;\n"
	"layout(location = 1) out vec4 o_X;\n"
	"void main() {\n"
	"  vec3 kc = texture(t_rgb, f_uv).rgb;\n"
	"  vec3 wc = clamp((kc - u_woff) / u_wgain, 0.0, 1.0);\n"
	"  bool ok = f_occl < 0.5;\n"
	"  o_gray = ok ? round(dot(wc * 255.0, vec3(0.299, 0.587, 0.114))) : -1.0;\n"
	"  o_X = vec4(f_X, ok ? 1.0 : 0.0);\n"
	"}\n";

static const char BODY_COMP_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 14) uniform sampler2D t_sgray;\n"
	"layout(binding = 15) uniform sampler2D t_wgray;\n"
	"layout(binding = 2, r32f) uniform writeonly image2D i_synf;\n" /* image units 0..7 are the limit here; the flow module rebinds 2 itself */
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  float g = texelFetch(t_sgray, p, 0).r;\n"
	"  imageStore(i_synf, p, vec4(g >= 0.0 ? g : texelFetch(t_wgray, p, 0).r, 0.0, 0.0, 0.0));\n"
	"}\n";

static const char BODY_GN_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 14) uniform sampler2D t_lflow;\n"
	"layout(binding = 15) uniform sampler2D t_sX;\n"
	"layout(std430, binding = 0) writeonly buffer Part { float part[]; };\n"
	"shared float sh[256 * 29];\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  uint li = gl_LocalInvocationIndex;\n"
	"  float v[29];\n"
	"  for (int k = 0; k < 29; k++) v[k] = 0.0;\n"
	"  vec4 F = texelFetch(t_lflow, p, 0), X = texelFetch(t_sX, p, 0);\n"
	"  float a = length(F.xy);\n"
	"  if (F.w > 0.5 && X.w > 0.5 && a < 16.0) {\n"
	"    vec3 c = u_wR * X.xyz + u_wt;\n"
	"    float iz = 1.0 / c.z;\n"
	"    vec3 du = vec3(u_wK.x * iz, 0.0, -u_wK.x * c.x * iz * iz);\n"
	"    vec3 dv = vec3(0.0, u_wK.y * iz, -u_wK.y * c.y * iz * iz);\n"
	"    vec3 cw0 = vec3(0.0, c.z, -c.y), cw1 = vec3(-c.z, 0.0, c.x), cw2 = vec3(c.y, -c.x, 0.0);\n" /* columns of -[c]x */
	"    float J[12] = float[12](dot(du, cw0), dot(du, cw1), dot(du, cw2), du.x, du.y, du.z,\n"
	"                            dot(dv, cw0), dot(dv, cw1), dot(dv, cw2), dv.x, dv.y, dv.z);\n"
	"    float w = a < 1.5 ? 1.0 : 1.5 / a;\n"
	"    int k = 0;\n"
	"    for (int i = 0; i < 6; i++)\n"
	"      for (int j = i; j < 6; j++) v[k++] = w * (J[i] * J[j] + J[6 + i] * J[6 + j]);\n"
	"    for (int i = 0; i < 6; i++) v[21 + i] = w * (J[i] * F.x + J[6 + i] * F.y);\n"
	"    v[27] = a;\n"
	"    v[28] = 1.0;\n"
	"  }\n"
	"  for (int k = 0; k < 29; k++) sh[li * 29u + uint(k)] = v[k];\n"
	"  barrier();\n"
	"  for (uint s = 128u; s > 0u; s >>= 1) {\n"
	"    if (li < s) for (int k = 0; k < 29; k++) sh[li * 29u + uint(k)] += sh[(li + s) * 29u + uint(k)];\n"
	"    barrier();\n"
	"  }\n"
	"  if (li == 0u) {\n"
	"    uint g = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;\n"
	"    for (int k = 0; k < 29; k++) part[g * 29u + uint(k)] = sh[k];\n"
	"  }\n"
	"}\n";

/* ==== velocity field in room coordinates ========================================================
 * sflow.comp, per depth pixel p seen by the RGB camera: Kinect RGB flow (newest -> previous) takes its
 * RGB position c back to cp; the previous frame's winner map gives the depth pixel that saw cp, with a
 * sub-pixel correction through the local registration slope; both 3D points go into the room frame
 * with that moment's rotation (u_Rnow, u_Rprev: rows = room right, up, forward in camera coordinates),
 * v = (W_now - W_prev) / frame period (mm/s). Also writes the camera-frame position for the cells. */
static const char BODY_SFLOW_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 13) uniform usampler2D t_dprev;\n"
	"layout(binding = 14) uniform usampler2D t_winprev;\n"
	"layout(binding = 15) uniform sampler2D t_kflow;\n"
	"layout(binding = 1, rgba32f) uniform writeonly image2D i_vel;\n"
	"layout(binding = 4, rgba32f) uniform writeonly image2D i_pos;\n"
	"uniform mat3 u_Rnow;\n"
	"uniform mat3 u_Rprev;\n"
	"uniform float u_idt;\n"
	"vec3 kworldf(vec2 p, float z) { return vec3((p.x - 320.0) * u_k * z, (p.y - 240.0) * u_k * z, z); }\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  Pix q = pix(p);\n"
	"  vec4 vout = vec4(0.0), pout = vec4(0.0);\n"
	"  if (q.ok && q.inb) {\n"
	"    vec3 Xn = kworld(p, q.wz);\n"
	"    pout = vec4(Xn, 1.0);\n"
	"    float z = float(q.wz), zm = float(texelFetch(t_zmin, ivec2(q.cx, q.cy), 0).r);\n"
	"    vec2 c = vec2(float(q.s) / 256.0, float(q.cy) + 0.5);\n"
	"    vec4 F = texture(t_kflow, c / vec2(640.0, 480.0));\n"
	"    if (z <= zm + max(50.0, 3.0 * z * z / 348000.0) && F.w > 0.999) {\n" /* RGB sees it; all 4 flow taps valid */
	"      vec2 cp = c + F.xy;\n"
	"      ivec2 ci = ivec2(floor(cp));\n"
	"      float best = 4.0;\n"
	"      ivec2 bq = ivec2(-1);\n"
	"      vec2 bc = vec2(0.0);\n"
	"      uint bz = 0u;\n"
	"      for (int dy = -1; dy <= 1; dy++)\n"
	"        for (int dx = -1; dx <= 1; dx++) {\n"
	"          ivec2 cc = ci + ivec2(dx, dy);\n"
	"          if (any(lessThan(cc, ivec2(0))) || any(greaterThanEqual(cc, ivec2(640, 480)))) continue;\n"
	"          uint w = texelFetch(t_winprev, cc, 0).r;\n"
	"          if (w == 0xFFFFFFFFu) continue;\n"
	"          ivec2 qp = ivec2(int(w % 640u), int(w / 640u));\n"
	"          uint zq = texelFetch(t_dprev, qp, 0).r;\n"
	"          if (zq == 0u || zq >= 10000u) continue;\n"
	"          ivec2 rg = texelFetch(t_reg, qp, 0).xy;\n"
	"          vec2 cq = vec2(float(rg.x + texelFetch(t_shift, ivec2(int(zq), 0), 0).r) / 256.0, float(rg.y) + 0.5);\n"
	"          float d2 = dot(cq - cp, cq - cp);\n"
	"          if (d2 < best) { best = d2; bq = qp; bc = cq; bz = zq; }\n"
	"        }\n"
	"      if (bq.x >= 0) {\n"
	"        ivec2 a = min(bq + ivec2(1, 0), ivec2(639, 479)), b = max(bq - ivec2(1, 0), ivec2(0));\n"
	"        ivec2 u = min(bq + ivec2(0, 4), ivec2(639, 479)), d = max(bq - ivec2(0, 4), ivec2(0));\n"
	"        float sx = float(texelFetch(t_reg, a, 0).x - texelFetch(t_reg, b, 0).x) / (256.0 * float(max(a.x - b.x, 1)));\n"
	"        float sy = float(texelFetch(t_reg, u, 0).y - texelFetch(t_reg, d, 0).y) / float(max(u.y - d.y, 1));\n"
	"        vec2 pq = vec2(bq) + (cp - bc) / clamp(vec2(sx, sy), vec2(0.5), vec2(2.0));\n"
	"        vec3 v = (u_Rnow * Xn - u_Rprev * kworldf(pq, float(bz))) * u_idt;\n"
	"        if (length(v) < 5000.0) vout = vec4(v, 1.0);\n"
	"      }\n"
	"    }\n"
	"  }\n"
	"  imageStore(i_vel, p, vout);\n"
	"  imageStore(i_pos, p, pout);\n"
	"}\n";

/* 80x60 cells of 8x8 pixels: mean room velocity (w = valid count), mean camera-frame position (w = count) */
static const char BODY_CELLS_COMP[] =
	"layout(local_size_x = 8, local_size_y = 8) in;\n"
	"layout(binding = 14) uniform sampler2D t_vel;\n"
	"layout(binding = 15) uniform sampler2D t_pos;\n"
	"layout(binding = 1, rgba32f) uniform writeonly image2D i_cv;\n"
	"layout(binding = 4, rgba32f) uniform writeonly image2D i_cp;\n"
	"void main() {\n"
	"  ivec2 c = ivec2(gl_GlobalInvocationID.xy);\n"
	"  if (c.x >= 80 || c.y >= 60) return;\n"
	"  vec3 sv = vec3(0.0), sp = vec3(0.0);\n"
	"  float nv = 0.0, np = 0.0;\n"
	"  for (int j = 0; j < 8; j++)\n"
	"    for (int i = 0; i < 8; i++) {\n"
	"      ivec2 p = c * 8 + ivec2(i, j);\n"
	"      vec4 v = texelFetch(t_vel, p, 0), x = texelFetch(t_pos, p, 0);\n"
	"      if (v.w > 0.5) { sv += v.xyz; nv += 1.0; }\n"
	"      if (x.w > 0.5) { sp += x.xyz; np += 1.0; }\n"
	"    }\n"
	"  imageStore(i_cv, c, vec4(nv > 0.0 ? sv / nv : vec3(0.0), nv));\n"
	"  imageStore(i_cp, c, vec4(np > 0.0 ? sp / np : vec3(0.0), np));\n"
	"}\n";

/* arrows: one GL_LINES segment per cell, from its mean position along its velocity * u_ascale seconds */
static const char BODY_ARROW_VERT[] =
	"layout(binding = 14) uniform sampler2D t_cv;\n"
	"layout(binding = 15) uniform sampler2D t_cp;\n"
	"uniform mat3 u_Rnow;\n"
	"uniform float u_vmin;\n"
	"uniform float u_vz2;\n"
	"uniform float u_ascale;\n"
	"out vec3 a_col;\n"
	"void main() {\n"
	"  int id = gl_VertexID >> 1, e = gl_VertexID & 1;\n"
	"  ivec2 c = ivec2(id % 80, id / 80);\n"
	"  vec4 v = texelFetch(t_cv, c, 0), x = texelFetch(t_cp, c, 0);\n"
	"  float s = length(v.xyz);\n"
	"  vec3 P = x.xyz + float(e) * (transpose(u_Rnow) * v.xyz) * u_ascale;\n" /* room -> camera */
	"  gl_Position = (v.w >= 8.0 && x.w >= 8.0 && s >= u_vmin + u_vz2 * x.z * x.z * 1e-6) ? u_mvp * vec4(P.x, -P.y, -P.z, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);\n"
	"  a_col = mix(vec3(0.1, 0.6, 1.0), vec3(1.0, 0.15, 0.1), clamp(s / 1000.0, 0.0, 1.0));\n"
	"}\n";

static const char BODY_ARROW_FRAG[] =
	"in vec3 a_col;\n"
	"out vec4 o;\n"
	"void main() { o = vec4(a_col, 1.0); }\n";

/* frame generation (the FSR 3 optical flow project's mid-instant field): a pixel x at the middle instant came
 * from x + v/2 in the older frame and goes to x - v/2 in the newer (framegen.c: va + vb = v, half each way);
 * v by bilinear interpolation between block centres = overlapping tent windows, so no block edge shows */
static const char BODY_FG_COMP[] =
	"layout(local_size_x = 16, local_size_y = 16) in;\n"
	"layout(binding = 13) uniform sampler2D t_older;\n"
	"layout(binding = 14) uniform sampler2D t_newer;\n"
	"layout(binding = 15) uniform sampler2D t_mid;\n"
	"layout(binding = 1, rgba8) uniform writeonly image2D i_gen;\n"
	"layout(location = 0) uniform float u_sign; /* 1; -1 only in the test control */\n"
	"void main() {\n"
	"  ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
	"  vec2 sz = vec2(640.0, 480.0), x = vec2(p) + 0.5;\n"
	"  vec2 v = u_sign * texture(t_mid, x / sz).xy;\n"
	"  vec3 a = texture(t_older, (x + 0.5 * v) / sz).rgb, b = texture(t_newer, (x - 0.5 * v) / sz).rgb;\n"
	"  imageStore(i_gen, p, vec4(0.5 * (a + b), 1.0));\n"
	"}\n";

static const struct { const char *file; GLenum stage; const char *body; } STAGES[] = {
	{"mesh_a.vert", GL_VERTEX_SHADER, BODY_MESH_A_VERT},
	{"mesh_a.geom", GL_GEOMETRY_SHADER, BODY_MESH_A_GEOM},
	{"mesh_b.vert", GL_VERTEX_SHADER, BODY_MESH_B_VERT},
	{"mesh.frag", GL_FRAGMENT_SHADER, BODY_MESH_FRAG},
	{"scatter.comp", GL_COMPUTE_SHADER, BODY_SCATTER_COMP},
	{"check.comp", GL_COMPUTE_SHADER, BODY_CHECK_COMP},
	{"pip.vert", GL_VERTEX_SHADER, BODY_PIP_VERT},
	{"pip.frag", GL_FRAGMENT_SHADER, BODY_PIP_FRAG},
	{"winner.comp", GL_COMPUTE_SHADER, BODY_WINNER_COMP},
	{"pmap.comp", GL_COMPUTE_SHADER, BODY_PMAP_COMP},
	{"wz.comp", GL_COMPUTE_SHADER, BODY_WZ_COMP},
	{"dfill.comp", GL_COMPUTE_SHADER, BODY_DFILL_COMP},
	{"syn.frag", GL_FRAGMENT_SHADER, BODY_SYN_FRAG},
	{"comp.comp", GL_COMPUTE_SHADER, BODY_COMP_COMP},
	{"gn.comp", GL_COMPUTE_SHADER, BODY_GN_COMP},
	{"sflow.comp", GL_COMPUTE_SHADER, BODY_SFLOW_COMP},
	{"cells.comp", GL_COMPUTE_SHADER, BODY_CELLS_COMP},
	{"arrow.vert", GL_VERTEX_SHADER, BODY_ARROW_VERT},
	{"arrow.frag", GL_FRAGMENT_SHADER, BODY_ARROW_FRAG},
	{"fg.comp", GL_COMPUTE_SHADER, BODY_FG_COMP},
};
enum { S_AV, S_AG, S_BV, S_FR, S_SC, S_CK, S_PV, S_PF, S_WN, S_PM, S_WZ, S_DF, S_SY, S_CO, S_GN, S_SF, S_CE, S_AWV, S_AWF, S_FG, NSTAGE };

static int dump_shaders(const char *dir)
{
	for (int i = 0; i < NSTAGE; i++) {
		char path[1024];
		snprintf(path, sizeof path, "%s/%s", dir, STAGES[i].file);
		FILE *f = fopen(path, "w");
		if (!f || fputs(GLSL_COMMON, f) < 0 || fputs(STAGES[i].body, f) < 0 || fclose(f)) {
			perror(path);
			return 1;
		}
	}
	return flow_dump_shaders(dir);
}

/* ==== Kinect ==================================================================================== */
static freenect_context *fctx;
static freenect_device *fdev;
static tribuf tb_depth, tb_rgb;
static double k_depth_t[3], k_rgb_t[3]; /* arrival time (CLOCK_MONOTONIC) per triple-buffer slot */
static pthread_t kthr;
static atomic_int k_die, k_restart_video;
static atomic_uint k_ndepth, k_nrgb, k_bufmiss;

static double mono_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void depth_cb(freenect_device *d, void *buf, uint32_t ts)
{
	(void)ts;
	if ((uint8_t *)buf != tb_depth.buf[tb_depth.back]) atomic_fetch_add(&k_bufmiss, 1);
	k_depth_t[tb_depth.back] = mono_s();
	freenect_set_depth_buffer(d, tb_publish(&tb_depth));
	atomic_fetch_add_explicit(&k_ndepth, 1, memory_order_relaxed);
}

static void video_cb(freenect_device *d, void *buf, uint32_t ts)
{
	(void)ts;
	if ((uint8_t *)buf != tb_rgb.buf[tb_rgb.back]) atomic_fetch_add(&k_bufmiss, 1);
	k_rgb_t[tb_rgb.back] = mono_s();
	freenect_set_video_buffer(d, tb_publish(&tb_rgb));
	atomic_fetch_add_explicit(&k_nrgb, 1, memory_order_relaxed);
}

static void *kinect_thread(void *arg)
{
	(void)arg;
	while (!atomic_load(&k_die)) {
		if (atomic_exchange(&k_restart_video, 0)) {
			freenect_stop_video(fdev);
			freenect_start_video(fdev);
		}
		struct timeval tv = {0, 50000};
		if (freenect_process_events_timeout(fctx, &tv) < 0) {
			fprintf(stderr, "kinect3d: freenect_process_events failed\n");
			break;
		}
	}
	return NULL;
}

static int kinect_open(void)
{
	if (freenect_init(&fctx, NULL) < 0) { fprintf(stderr, "kinect3d: freenect_init failed\n"); return -1; }
	freenect_set_log_level(fctx, FREENECT_LOG_WARNING);
	freenect_select_subdevices(fctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));
	if (freenect_num_devices(fctx) < 1) { fprintf(stderr, "kinect3d: no Kinect found\n"); return -1; }
	if (freenect_open_device(fctx, &fdev, 0) < 0) {
		fprintf(stderr, "kinect3d: cannot open the Kinect (another program may hold it)\n");
		return -1;
	}
	freenect_frame_mode dm = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM);
	freenect_frame_mode vm = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB);
	if (!dm.is_valid || !vm.is_valid || dm.bytes != NPIX * 2 || vm.bytes != NPIX * 3) {
		fprintf(stderr, "kinect3d: unexpected modes: depth %dx%d %d B, video %dx%d %d B\n",
			dm.width, dm.height, dm.bytes, vm.width, vm.height, vm.bytes);
		return -1;
	}
	if (tb_init(&tb_depth, NPIX * 2) || tb_init(&tb_rgb, NPIX * 3)) { fprintf(stderr, "kinect3d: out of memory\n"); return -1; }
	freenect_set_depth_mode(fdev, dm);
	freenect_set_video_mode(fdev, vm);
	freenect_set_depth_buffer(fdev, tb_depth.buf[tb_depth.back]);
	freenect_set_video_buffer(fdev, tb_rgb.buf[tb_rgb.back]);
	freenect_set_depth_callback(fdev, depth_cb);
	freenect_set_video_callback(fdev, video_cb);
	freenect_set_led(fdev, LED_GREEN);
	if (freenect_start_depth(fdev) < 0 || freenect_start_video(fdev) < 0) {
		fprintf(stderr, "kinect3d: cannot start the streams\n");
		return -1;
	}
	if (pthread_create(&kthr, NULL, kinect_thread, NULL)) { fprintf(stderr, "kinect3d: pthread_create failed\n"); return -1; }
	return 0;
}

static void kinect_close(void)
{
	atomic_store(&k_die, 1);
	pthread_join(kthr, NULL);
	freenect_stop_video(fdev);
	freenect_stop_depth(fdev);
	freenect_set_led(fdev, LED_BLINK_GREEN);
	freenect_close_device(fdev);
	freenect_shutdown(fctx);
}

/* ==== matrices (column-major) =================================================================== */
typedef struct { float m[16]; } mat4;

static mat4 mid(void) { mat4 r = {{0}}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f; return r; }
static mat4 mmul(mat4 a, mat4 b)
{
	mat4 r;
	for (int c = 0; c < 4; c++)
		for (int w = 0; w < 4; w++)
			r.m[c * 4 + w] = a.m[w] * b.m[c * 4] + a.m[4 + w] * b.m[c * 4 + 1] + a.m[8 + w] * b.m[c * 4 + 2] + a.m[12 + w] * b.m[c * 4 + 3];
	return r;
}
static mat4 mtrans(float x, float y, float z) { mat4 r = mid(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static mat4 mrotx(float a) { mat4 r = mid(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
static mat4 mroty(float a) { mat4 r = mid(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
static mat4 mpersp(float fovy, float aspect, float n, float f)
{
	mat4 r = {{0}};
	float t = 1.0f / tanf(fovy * 0.5f);
	r.m[0] = t / aspect; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1.0f; r.m[14] = 2.0f * f * n / (n - f);
	return r;
}

/* ==== GL state ================================================================================== */
static GLuint p_mesh[2], p_scatter, p_check, vao[2], ebo;
static GLuint t_depth, t_reg, t_shift, t_zmin, t_rgb, t_out;
static GLint loc_mvp[2], loc_k[2], loc_cull[2], loc_occl[2], loc_tshift[2], loc_rcam[2];
static int g_geo = 1;          /* geometry follows the instant of the colour on screen (T) */
static float g_tshift;         /* this draw's shift, s */
static double g_tshift_sum, g_t_rgb;
static unsigned g_tshift_n;
static float g_k, g_cull = 0.04f;
static int g_occl = 1, g_method = DEFAULT_METHOD;
static GLuint t_win, t_wz, t_pmap, p_win, p_pmap, p_wz, t_draw, t_dtemp, p_dfill;
static int g_dfilter = 1;
static GLuint t_dprev, t_winprev, t_vel, t_pos, t_cv, t_cp, p_sflow, p_cells, p_arrow;
static GLuint p_syn, p_comp, p_gn, fbo_syn, t_sgray, t_sX, t_synf, ssbo_part;
static flowcam *g_flock;
static int g_lock, g_lock_measure;
static struct { unsigned n, updates; double mean_r, step_deg, step_mm; } g_lk;
/* webcam calibration (tools/kcalib.cpp): X_webcam = R X_kinect + t (mm), OpenCV pinhole + k1 k2,
 * webcam colour mapped to Kinect colour by gain/offset per channel */
static struct { int ok; float K[4], D[2], R[9], t[3], gain[3], off[3]; char path[512]; } g_cal;
static unsigned gl_errors;
static struct { float yaw, pitch, dist, px, py, pz; } cam;
static double mx_last, my_last;
static int drag; /* 1 orbit, 2 pan */
static int tilt_deg;
static GLuint p_pip, t_wrgb;
static GLint loc_pip_rect, loc_pip_mode, loc_pip_box, loc_pip_boxcol, loc_pip_feat, loc_pip_nfeat, loc_pip_nmem;
static int g_pip = 1; /* 0 off, 1 camera feeds, 2 flow */
static webcam *g_cam;
static flowcam *g_fk, *g_fw;

static void APIENTRY gl_debug(GLenum src, GLenum type, GLuint id, GLenum sev, GLsizei len, const GLchar *msg, const void *user)
{
	(void)src; (void)id; (void)len; (void)user;
	if (type == GL_DEBUG_TYPE_ERROR || sev == GL_DEBUG_SEVERITY_HIGH) {
		if (gl_errors++ < 10) fprintf(stderr, "kinect3d: GL: %s\n", msg);
	}
}

static GLuint compile(int stage)
{
	const char *src[2] = {GLSL_COMMON, STAGES[stage].body};
	GLuint s = glCreateShader(STAGES[stage].stage);
	GLint ok = 0, len = 0;
	glShaderSource(s, 2, src, NULL);
	glCompileShader(s);
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
	if (!ok || len > 1) { /* warnings are errors */
		char log[4096];
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		fprintf(stderr, "kinect3d: shader %s %s:\n%s\n", STAGES[stage].file, ok ? "has diagnostics" : "failed", log);
		exit(2);
	}
	return s;
}

static GLuint build_program(int a, int b, int c)
{
	GLuint p = glCreateProgram();
	int st[3] = {a, b, c};
	GLint ok = 0, len = 0;
	for (int i = 0; i < 3; i++)
		if (st[i] >= 0) glAttachShader(p, compile(st[i]));
	glLinkProgram(p);
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
	if (!ok || len > 1) {
		char log[4096];
		glGetProgramInfoLog(p, sizeof log, NULL, log);
		fprintf(stderr, "kinect3d: link %s:\n%s\n", ok ? "has diagnostics" : "failed", log);
		exit(2);
	}
	return p;
}

static GLuint tex2d(GLenum fmt, int w, int h, int linear)
{
	GLuint t;
	glCreateTextures(GL_TEXTURE_2D, 1, &t);
	glTextureStorage2D(t, 1, fmt, w, h);
	glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(t, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return t;
}

static void gl_setup(void)
{
	p_mesh[0] = build_program(S_AV, S_AG, S_FR);
	p_mesh[1] = build_program(S_BV, S_FR, -1);
	p_scatter = build_program(S_SC, -1, -1);
	p_check = build_program(S_CK, -1, -1);
	p_pip = build_program(S_PV, S_PF, -1);
	p_win = build_program(S_WN, -1, -1);
	p_pmap = build_program(S_PM, -1, -1);
	p_wz = build_program(S_WZ, -1, -1);
	p_dfill = build_program(S_DF, -1, -1);
	p_syn = build_program(S_AV, S_AG, S_SY);
	p_comp = build_program(S_CO, -1, -1);
	p_gn = build_program(S_GN, -1, -1);
	loc_pip_rect = glGetUniformLocation(p_pip, "u_rect");
	loc_pip_mode = glGetUniformLocation(p_pip, "u_mode");
	loc_pip_box = glGetUniformLocation(p_pip, "u_box");
	loc_pip_boxcol = glGetUniformLocation(p_pip, "u_boxcol");
	loc_pip_feat = glGetUniformLocation(p_pip, "u_feat");
	loc_pip_nfeat = glGetUniformLocation(p_pip, "u_nfeat");
	loc_pip_nmem = glGetUniformLocation(p_pip, "u_nmem");
	for (int m = 0; m < 2; m++) {
		loc_mvp[m] = glGetUniformLocation(p_mesh[m], "u_mvp");
		loc_tshift[m] = glGetUniformLocation(p_mesh[m], "u_tshift");
		loc_rcam[m] = glGetUniformLocation(p_mesh[m], "u_Rcam");
		loc_k[m] = glGetUniformLocation(p_mesh[m], "u_k");
		loc_cull[m] = glGetUniformLocation(p_mesh[m], "u_cull");
		loc_occl[m] = glGetUniformLocation(p_mesh[m], "u_occl");
	}
	t_depth = tex2d(GL_R16UI, W, H, 0);
	t_reg = tex2d(GL_RG32I, W, H, 0);
	t_shift = tex2d(GL_R32I, ZMAX, 1, 0);
	t_zmin = tex2d(GL_R32UI, W, H, 0);
	t_rgb = tex2d(GL_RGB8, W, H, 1);
	t_out = tex2d(GL_RGBA8, W, H, 0);
	t_wrgb = tex2d(GL_RGB8, W, H, 1);
	t_win = tex2d(GL_R32UI, W, H, 0);
	t_wz = tex2d(GL_R32UI, W / 2, H / 2, 0);
	t_pmap = tex2d(GL_RGBA32F, W, H, 0);
	glBindTextureUnit(6, t_win);
	glBindTextureUnit(7, t_wz);
	glBindTextureUnit(12, t_wrgb);
	glBindImageTexture(3, t_win, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(4, t_pmap, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(5, t_wz, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	t_draw = tex2d(GL_R16UI, W, H, 0);
	t_dtemp = tex2d(GL_RG32F, W, H, 0);
	glBindTextureUnit(13, t_draw);
	glBindImageTexture(6, t_depth, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16UI);
	glBindImageTexture(7, t_dtemp, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
	GLuint units[5] = {t_depth, t_reg, t_shift, t_zmin, t_rgb};
	for (GLuint u = 0; u < 5; u++) glBindTextureUnit(u, units[u]);
	glBindImageTexture(0, t_zmin, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(1, t_out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

	/* path A: index buffer with the same triangles, in the same order and winding, as path B */
	uint32_t *idx = malloc(sizeof(uint32_t) * NVERT);
	if (!idx) { fprintf(stderr, "kinect3d: out of memory\n"); exit(1); }
	for (uint32_t q = 0, o = 0; q < NQUAD; q++) {
		uint32_t v = (q / (W - 1)) * W + q % (W - 1);
		uint32_t c4[4] = {v, v + 1, v + W, v + W + 1};
		static const uint8_t pick[6] = {0, 1, 2, 1, 3, 2};
		for (int k = 0; k < 6; k++) idx[o++] = c4[pick[k]];
	}
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(uint32_t) * NVERT, idx, 0);
	free(idx);
	glCreateVertexArrays(2, vao);
	glVertexArrayElementBuffer(vao[0], ebo);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
}

static void upload_registration(const freenect_registration *reg)
{
	int32_t *t = malloc(sizeof(int32_t) * NPIX * 2);
	if (!t) { fprintf(stderr, "kinect3d: out of memory\n"); exit(1); }
	int32_t off = (int32_t)reg->reg_pad_info.start_lines * H; /* as freenect_map_rgb_to_depth */
	for (int i = 0; i < NPIX; i++) {
		t[2 * i] = reg->registration_table[i][0];
		t[2 * i + 1] = reg->registration_table[i][1] - off;
	}
	glTextureSubImage2D(t_reg, 0, 0, 0, W, H, GL_RG_INTEGER, GL_INT, t);
	glTextureSubImage2D(t_shift, 0, 0, 0, ZMAX, 1, GL_RED_INTEGER, GL_INT, reg->depth_to_rgb_shift);
	free(t);
	g_k = 2.0f * reg->zero_plane_info.reference_pixel_size / reg->zero_plane_info.reference_distance;
}

static unsigned g_depth_up, g_pmap_at; /* depth frames uploaded; the one the point map was last computed from */
static double g_t_depth; /* arrival time of the newest depth frame: the instant the point map is of */
static void upload_depth(const uint8_t *d)
{
	g_depth_up++;
	static const GLuint all = 0xFFFFFFFFu;
	if (t_dprev) { /* the previous filtered depth and winner map, for the velocity field */
		glCopyImageSubData(t_depth, GL_TEXTURE_2D, 0, 0, 0, 0, t_dprev, GL_TEXTURE_2D, 0, 0, 0, 0, W, H, 1);
		glCopyImageSubData(t_win, GL_TEXTURE_2D, 0, 0, 0, 0, t_winprev, GL_TEXTURE_2D, 0, 0, 0, 0, W, H, 1);
	}
	glTextureSubImage2D(t_draw, 0, 0, 0, W, H, GL_RED_INTEGER, GL_UNSIGNED_SHORT, d);
	glBindTextureUnit(13, t_draw); /* shared unit: every pass binds its own */
	glProgramUniform1i(p_dfill, glGetUniformLocation(p_dfill, "u_fill"), g_dfilter ? 16 : 0);
	glProgramUniform1f(p_dfill, glGetUniformLocation(p_dfill, "u_alpha"), g_dfilter ? 0.3f : 1.0f);
	glUseProgram(p_dfill);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
	glClearTexImage(t_zmin, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &all);
	glUseProgram(p_scatter);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glClearTexImage(t_win, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &all);
	glUseProgram(p_win);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	if (g_cal.ok) {
		glClearTexImage(t_wz, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &all);
		glUseProgram(p_wz);
		glDispatchCompute(W / 16, H / 16, 1);
		glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}
}

static void upload_rgb(const uint8_t *v) { glTextureSubImage2D(t_rgb, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, v); }

/* calibration uniforms into every program that projects into the webcam */
static void apply_calib(void)
{
	GLuint progs[6] = {p_mesh[0], p_mesh[1], p_pmap, p_wz, p_syn, p_gn};
	for (int i = 0; i < 6; i++) {
		GLuint p = progs[i];
		glProgramUniform1f(p, glGetUniformLocation(p, "u_k"), g_k);
		glProgramUniform1i(p, glGetUniformLocation(p, "u_wcal"), g_cal.ok);
		glProgramUniformMatrix3fv(p, glGetUniformLocation(p, "u_wR"), 1, GL_TRUE, g_cal.R); /* file is row-major */
		glProgramUniform3fv(p, glGetUniformLocation(p, "u_wt"), 1, g_cal.t);
		glProgramUniform4fv(p, glGetUniformLocation(p, "u_wK"), 1, g_cal.K);
		glProgramUniform2fv(p, glGetUniformLocation(p, "u_wd"), 1, g_cal.D);
		glProgramUniform3fv(p, glGetUniformLocation(p, "u_wgain"), 1, g_cal.gain);
		glProgramUniform3fv(p, glGetUniformLocation(p, "u_woff"), 1, g_cal.off);
	}
}

/* lines "K fx fy cx cy", "D k1 k2", "R r00 .. r22", "t tx ty tz", "gain r g b", "offset r g b"; # = comment */
static int load_calib(const char *path)
{
	FILE *f = fopen(path, "r");
	char line[512];
	unsigned seen = 0;
	if (!f) return -1;
	g_cal.gain[0] = g_cal.gain[1] = g_cal.gain[2] = 1.0f;
	while (fgets(line, sizeof line, f)) {
		float *v = NULL;
		int n = 0;
		char key[16];
		if (sscanf(line, "%15s", key) != 1 || key[0] == '#') continue;
		if (!strcmp(key, "K")) { v = g_cal.K; n = 4; seen |= 1; }
		else if (!strcmp(key, "D")) { v = g_cal.D; n = 2; seen |= 2; }
		else if (!strcmp(key, "R")) { v = g_cal.R; n = 9; seen |= 4; }
		else if (!strcmp(key, "t")) { v = g_cal.t; n = 3; seen |= 8; }
		else if (!strcmp(key, "gain")) { v = g_cal.gain; n = 3; }
		else if (!strcmp(key, "offset")) { v = g_cal.off; n = 3; }
		else continue;
		const char *s = line + strlen(key);
		for (int i = 0; i < n; i++) {
			char *e;
			v[i] = strtof(s, &e);
			if (e == s) { fclose(f); fprintf(stderr, "kinect3d: %s: short line for %s\n", path, key); return -1; }
			s = e;
		}
	}
	fclose(f);
	if (seen != 15) { fprintf(stderr, "kinect3d: %s: missing K, D, R or t\n", path); return -1; }
	snprintf(g_cal.path, sizeof g_cal.path, "%s", path);
	g_cal.ok = 1;
	return 0;
}



/* ==== velocity field in room coordinates (host) ================================================== */

static int g_arrows = 1;
static fsrcam *g_fsk, *g_fsw; /* the user's FSR 3 optical flow on the Kinect RGB and the webcam */
/* the face tracker: VisionTrack's region tracker per camera (vtface.c), kept on the face by YuNet */
static vtface *g_vtf[2];
static vtf_out g_fo[2];
static int g_facetrack = 1;
static unsigned g_fstate[2][6], g_fres[2];
/* the face in the room: Kinect point under the Kinect box's centre, room frame (mm); velocity = least-squares slope over 0.3 s */
static struct {
	int have;
	double t[16], p[16][3], pos[3], vel[3], t_last;
	unsigned n, k, points, misses;
} g_facep;
static GLuint t_fsk, t_fsw;
static int g_fsrc = 1;        /* velocity field from: 1 FSR (default), 0 GPU LK */
static float g_fsbuf[80 * 60 * 4];
static struct {
	double acc[3];          /* latest raw accelerometer, m/s^2 */
	double up[3];           /* up in camera coordinates, filtered */
	double X[3];            /* room right (wall-aligned) in camera coordinates */
	int sx, sz, ok, have_up, moving;
	double ceil_mm, wall_left, wall_right, wall_ahead, ceil_angle;
	unsigned ceil_n, wall_n;
	float Rnow[9], Rprev[9]; /* camera -> room, rows: right, up, forward */
} g_room = {.up = {0, -1, 0}, .X = {1, 0, 0}, .sx = 1, .sz = 1, .ceil_angle = NAN, .wall_left = NAN, .wall_right = NAN, .wall_ahead = NAN};
static float g_cvb[80 * 60 * 4], g_cpb[80 * 60 * 4];
static struct { unsigned n; double pos[3], vel[3]; } g_mov;

static void v3norm(double v[3]) { double n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (n > 0) for (int i = 0; i < 3; i++) v[i] /= n; }
static double v3dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static void v3cross(const double a[3], const double b[3], double o[3])
{
	o[0] = a[1] * b[2] - a[2] * b[1];
	o[1] = a[2] * b[0] - a[0] * b[2];
	o[2] = a[0] * b[1] - a[1] * b[0];
}

/* accelerometer -> up in camera coordinates (camera: x right, y down, z forward). The y mapping follows
 * the reading at rest (+9.28 m/s^2 with the camera's y axis pointing down); x and z signs are chosen by
 * the ceiling in room_geometry() */
static void room_accel(void)
{
	freenect_raw_tilt_state *st;
	if (freenect_update_tilt_state(fdev) < 0 || !(st = freenect_get_tilt_state(fdev))) return;
	freenect_get_mks_accel(st, &g_room.acc[0], &g_room.acc[1], &g_room.acc[2]);
	g_room.moving = freenect_get_tilt_status(st) == TILT_STATUS_MOVING;
	double u[3] = {g_room.sx * g_room.acc[0], -g_room.acc[1], g_room.sz * g_room.acc[2]};
	v3norm(u);
	for (int i = 0; i < 3; i++) g_room.up[i] = g_room.have_up ? 0.8 * g_room.up[i] + 0.2 * u[i] : u[i];
	v3norm(g_room.up);
	g_room.have_up = 1;
}

/* rows of camera -> room: right (X re-orthogonalised to the current up), up, forward = up x right */
static void room_rotation(float R[9])
{
	double x[3] = {g_room.X[0], g_room.X[1], g_room.X[2]}, f[3];
	const double k = v3dot(x, g_room.up);
	for (int i = 0; i < 3; i++) x[i] -= k * g_room.up[i];
	v3norm(x);
	v3cross(g_room.up, x, f);
	for (int i = 0; i < 3; i++) { R[i] = (float)x[i]; R[3 + i] = (float)g_room.up[i]; R[6 + i] = (float)f[i]; }
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* room geometry from one depth frame on a 160x120 grid: normals from neighbour cross products; ceiling =
 * normals facing down from above the camera (fixes the accelerometer's x/z signs, gives the ceiling height
 * and the angle between gravity and the ceiling normal); walls = vertical-plane normals folded mod 90 deg
 * (room yaw), distances from plane-offset histograms */
static void room_geometry(const uint16_t *d)
{
	enum { S = 4, GW = W / S, GH = H / S, NB = 240 };
	static double P[GH][GW][3], N[GH][GW][3];
	static uint8_t okp[GH][GW], okn[GH][GW];
	for (int gy = 0; gy < GH; gy++)
		for (int gx = 0; gx < GW; gx++) {
			const double z = d[gy * S * W + gx * S];
			okp[gy][gx] = z > 300 && z < 8000;
			P[gy][gx][0] = (gx * S - 320) * g_k * z;
			P[gy][gx][1] = (gy * S - 240) * g_k * z;
			P[gy][gx][2] = z;
			okn[gy][gx] = 0;
		}
	for (int gy = 1; gy < GH - 1; gy++)
		for (int gx = 1; gx < GW - 1; gx++) {
			if (!okp[gy][gx] || !okp[gy][gx - 1] || !okp[gy][gx + 1] || !okp[gy - 1][gx] || !okp[gy + 1][gx]) continue;
			const double z = P[gy][gx][2];
			if (fabs(P[gy][gx + 1][2] - P[gy][gx - 1][2]) > 0.1 * z || fabs(P[gy + 1][gx][2] - P[gy - 1][gx][2]) > 0.1 * z) continue;
			double a[3], b[3], *n = N[gy][gx];
			for (int i = 0; i < 3; i++) { a[i] = P[gy][gx + 1][i] - P[gy][gx - 1][i]; b[i] = P[gy + 1][gx][i] - P[gy - 1][gx][i]; }
			v3cross(a, b, n);
			v3norm(n);
			if (v3dot(n, P[gy][gx]) > 0) for (int i = 0; i < 3; i++) n[i] = -n[i]; /* face the camera */
			okn[gy][gx] = 1;
		}
	/* ceiling: the accelerometer sign pair whose up gathers the most down-facing normals above the camera */
	const double c12 = cos(12.0 * M_PI / 180.0);
	unsigned bestn = 0, curn = 0;
	int bsx = g_room.sx, bsz = g_room.sz;
	for (int sx = -1; sx <= 1; sx += 2)
		for (int sz = -1; sz <= 1; sz += 2) {
			double u[3] = {sx * g_room.acc[0], -g_room.acc[1], sz * g_room.acc[2]};
			v3norm(u);
			unsigned n = 0;
			for (int gy = 0; gy < GH; gy++)
				for (int gx = 0; gx < GW; gx++)
					n += okn[gy][gx] && -v3dot(N[gy][gx], u) > c12 && v3dot(P[gy][gx], u) > 300;
			if (sx == g_room.sx && sz == g_room.sz) curn = n;
			if (n > bestn) { bestn = n; bsx = sx; bsz = sz; }
		}
	g_room.ceil_n = bestn;
	if (bestn >= 50 && (bsx != g_room.sx || bsz != g_room.sz) && bestn > curn + curn / 4) { /* clear majority only: the x component is small */
		g_room.sx = bsx;
		g_room.sz = bsz;
		g_room.have_up = 0; /* restart the filter with the corrected mapping */
		room_accel();
	}
	double ua[3] = {g_room.sx * g_room.acc[0], -g_room.acc[1], g_room.sz * g_room.acc[2]}, ug[3] = {0, 0, 0};
	v3norm(ua);
	static double hs[GW * GH];
	unsigned nh = 0;
	for (int gy = 0; gy < GH; gy++)
		for (int gx = 0; gx < GW; gx++)
			if (okn[gy][gx] && -v3dot(N[gy][gx], ua) > c12 && v3dot(P[gy][gx], ua) > 300)
				for (int i = 0; i < 3; i++) ug[i] -= N[gy][gx][i];
	v3norm(ug);
	if (bestn >= 50) {
		g_room.ceil_angle = acos(fmin(1.0, v3dot(ug, ua))) * 180.0 / M_PI;
		for (int gy = 0; gy < GH; gy++)
			for (int gx = 0; gx < GW; gx++)
				if (okn[gy][gx] && -v3dot(N[gy][gx], ug) > c12 && v3dot(P[gy][gx], ug) > 300) hs[nh++] = v3dot(P[gy][gx], ug);
		qsort(hs, nh, sizeof(double), cmp_d);
		g_room.ceil_mm = nh ? hs[nh / 2] : NAN;
	}
	/* walls: vertical-plane normals, horizontal angle folded mod 90 deg; the peak is the room's yaw */
	const double *up = g_room.up;
	double e1[3] = {1, 0, 0}, e2[3];
	const double k = v3dot(e1, up);
	for (int i = 0; i < 3; i++) e1[i] -= k * up[i];
	v3norm(e1);
	v3cross(up, e1, e2);
	unsigned hist[90] = {0}, nw = 0;
	for (int gy = 0; gy < GH; gy++)
		for (int gx = 0; gx < GW; gx++) {
			if (!okn[gy][gx] || fabs(v3dot(N[gy][gx], up)) > 0.15) continue;
			double a = atan2(v3dot(N[gy][gx], e2), v3dot(N[gy][gx], e1)) * 180.0 / M_PI;
			a = fmod(a + 360.0, 90.0);
			hist[(int)a % 90]++;
			nw++;
		}
	g_room.wall_n = nw;
	if (nw >= 50) {
		int bb = 0;
		unsigned bv = 0;
		for (int b = 0; b < 90; b++) {
			unsigned v = hist[(b + 89) % 90] + hist[b] + hist[(b + 1) % 90];
			if (v > bv) { bv = v; bb = b; }
		}
		double yaw = bb + 0.5;
		if (yaw > 45) yaw -= 90; /* the Manhattan rotation closest to the camera's right */
		const double c = cos(yaw * M_PI / 180.0), s = sin(yaw * M_PI / 180.0);
		for (int i = 0; i < 3; i++) g_room.X[i] = c * e1[i] + s * e2[i];
		double f[3];
		v3cross(up, g_room.X, f);
		/* plane offsets: 50 mm bins over +-6 m along right (side walls) and forward (wall ahead) */
		static unsigned hx[NB], hf[NB];
		memset(hx, 0, sizeof hx);
		memset(hf, 0, sizeof hf);
		for (int gy = 0; gy < GH; gy++)
			for (int gx = 0; gx < GW; gx++) {
				if (!okn[gy][gx]) continue;
				if (fabs(v3dot(N[gy][gx], g_room.X)) > 0.9) { int b = (int)floor(v3dot(P[gy][gx], g_room.X) / 50.0) + NB / 2; if (b >= 0 && b < NB) hx[b]++; }
				if (v3dot(N[gy][gx], f) < -0.9) { int b = (int)floor(v3dot(P[gy][gx], f) / 50.0) + NB / 2; if (b >= 0 && b < NB) hf[b]++; }
			}
		unsigned bl = 0, br = 0, ba = 0;
		g_room.wall_left = g_room.wall_right = g_room.wall_ahead = NAN;
		for (int b = 0; b < NB; b++) {
			const double m = (b - NB / 2 + 0.5) * 50.0;
			if (m < 0 && hx[b] > bl && hx[b] >= 30) { bl = hx[b]; g_room.wall_left = m; }
			if (m > 0 && hx[b] > br && hx[b] >= 30) { br = hx[b]; g_room.wall_right = m; }
			if (m > 0 && hf[b] > ba) ba = hf[b];
		}
		for (int b = NB - 1; b >= 0; b--) /* farthest strong peak ahead: the nearest big plane is usually a person or a chair */
			if ((b - NB / 2) > 0 && hf[b] >= 30 && hf[b] * 4 >= ba) { g_room.wall_ahead = (b - NB / 2 + 0.5) * 50.0; break; }
		g_room.ok = bestn >= 50;
	}
}

static void sflow_setup(void)
{
	t_dprev = tex2d(GL_R16UI, W, H, 0);
	t_winprev = tex2d(GL_R32UI, W, H, 0);
	t_vel = tex2d(GL_RGBA32F, W, H, 0);
	glBindTextureUnit(16, t_vel); /* the mesh reads it to move vertices to the colour's instant */
	t_pos = tex2d(GL_RGBA32F, W, H, 0);
	t_cv = tex2d(GL_RGBA32F, 80, 60, 0);
	t_cp = tex2d(GL_RGBA32F, 80, 60, 0);
	p_sflow = build_program(S_SF, -1, -1);
	p_cells = build_program(S_CE, -1, -1);
	p_arrow = build_program(S_AWV, S_AWF, -1);
	room_rotation(g_room.Rnow);
	memcpy(g_room.Rprev, g_room.Rnow, sizeof g_room.Rnow);
}

/* one velocity field: newest depth vs previous depth through the newest Kinect RGB flow */
static void sflow_step(double period, GLuint flowtex)
{
	glProgramUniform1f(p_sflow, glGetUniformLocation(p_sflow, "u_k"), g_k);
	glProgramUniformMatrix3fv(p_sflow, glGetUniformLocation(p_sflow, "u_Rnow"), 1, GL_TRUE, g_room.Rnow);
	glProgramUniformMatrix3fv(p_sflow, glGetUniformLocation(p_sflow, "u_Rprev"), 1, GL_TRUE, g_room.Rprev);
	glProgramUniform1f(p_sflow, glGetUniformLocation(p_sflow, "u_idt"), (float)(1.0 / period));
	glBindTextureUnit(13, t_dprev);
	glBindTextureUnit(14, t_winprev);
	glBindTextureUnit(15, flowtex);
	glBindImageTexture(1, t_vel, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(4, t_pos, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glUseProgram(p_sflow);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindTextureUnit(14, t_vel);
	glBindTextureUnit(15, t_pos);
	glBindImageTexture(1, t_cv, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(4, t_cp, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glUseProgram(p_cells);
	glDispatchCompute(10, 8, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glGetTextureImage(t_cv, 0, GL_RGBA, GL_FLOAT, sizeof g_cvb, g_cvb);
	glGetTextureImage(t_cp, 0, GL_RGBA, GL_FLOAT, sizeof g_cpb, g_cpb);
	/* what moves: cells with >= 8 valid pixels and >= 200 mm/s, weighted by their valid count */
	double w = 0, ps[3] = {0, 0, 0}, vs[3] = {0, 0, 0};
	unsigned n = 0;
	for (int c = 0; c < 80 * 60; c++) {
		const float *v = g_cvb + 4 * c, *x = g_cpb + 4 * c;
		const float zm = x[2] * 1e-3f; /* threshold shaped like the measured noise floor (scene-stats 14:3x) */
		if (v[3] < 8 || x[3] < 8 || sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < 150.0f + 40.0f * zm * zm) continue;
		for (int i = 0; i < 3; i++) {
			ps[i] += v[3] * (g_room.Rnow[3 * i] * x[0] + g_room.Rnow[3 * i + 1] * x[1] + g_room.Rnow[3 * i + 2] * x[2]);
			vs[i] += v[3] * v[i];
		}
		w += v[3];
		n++;
	}
	if (n < 4) { n = 0; w = 0; } /* a lone cell is noise */
	g_mov.n = n;
	for (int i = 0; i < 3; i++) { g_mov.pos[i] = w > 0 ? ps[i] / w : 0; g_mov.vel[i] = w > 0 ? vs[i] / w : 0; }
}

static void draw_arrows(const mat4 *mvp)
{
	glProgramUniformMatrix4fv(p_arrow, glGetUniformLocation(p_arrow, "u_mvp"), 1, GL_FALSE, mvp->m);
	glProgramUniformMatrix3fv(p_arrow, glGetUniformLocation(p_arrow, "u_Rnow"), 1, GL_TRUE, g_room.Rnow);
	glProgramUniform1f(p_arrow, glGetUniformLocation(p_arrow, "u_vmin"), 150.0f);
	glProgramUniform1f(p_arrow, glGetUniformLocation(p_arrow, "u_vz2"), 40.0f);
	glProgramUniform1f(p_arrow, glGetUniformLocation(p_arrow, "u_ascale"), 0.25f);
	glBindTextureUnit(14, t_cv);
	glBindTextureUnit(15, t_cp);
	glUseProgram(p_arrow);
	glBindVertexArray(vao[1]);
	glDisable(GL_DEPTH_TEST);
	glDrawArrays(GL_LINES, 0, 80 * 60 * 2);
	glEnable(GL_DEPTH_TEST);
}

/* ==== frame generation: double both cameras' rate with the FSR 3 project's mid-instant field ==========
 * Real frame k arrives: cur -> prev on the GPU, k uploaded into cur, prev (= k-1, already on screen) stays
 * until the engine's field (k-1 -> k) arrives; then the generated frame (the instant half way) for half a
 * frame period; then k. Every image is shown about half a period: 2 images per camera frame, at the cost of
 * up to one period of display latency. Display only: the analysis passes read the real frames. */
typedef struct {
	GLuint cur, prev, gen, mid;
	int phase;        /* 0 prev (waiting for the field), 1 generated, 2 cur */
	double t_switch, period, t_cap_last, t_cap_prev;
	int show;         /* what the last fg_display put on screen: 0 prev, 1 generated, 2 cur */
	uint32_t seq;     /* frames pushed to this camera's engine = push number of cur */
	uint64_t last_id; /* content on screen: 2 seq for real frame seq, 2 seq - 1 for the one generated before it */
	unsigned shown, made, late, nofield; /* late: field came after the next frame; nofield: a period with none */
	double t_arr[4], lat_sum, lat_max; /* loop arrival of frame seq (ring) -> its first refresh */
	unsigned lat_n;
} fgcam;
static fgcam g_fgk, g_fgw;
static int g_fg = 1;
static GLuint p_fg, g_show_k, g_show_w;
static float g_rawbuf[80 * 60 * 2];

static void fg_setup(fgcam *f, GLuint cur)
{
	f->cur = cur;
	f->prev = tex2d(GL_RGB8, W, H, 1);
	f->gen = tex2d(GL_RGBA8, W, H, 1);
	f->mid = tex2d(GL_RG32F, 80, 60, 1);
	f->phase = 2;
}

/* before the new frame is uploaded into cur; t_cap = its capture time (period), now = loop time */
static void fg_new_frame(fgcam *f, double t_cap, double now)
{
	if (!f->gen) return;
	glCopyImageSubData(f->cur, GL_TEXTURE_2D, 0, 0, 0, 0, f->prev, GL_TEXTURE_2D, 0, 0, 0, 0, W, H, 1);
	const double dt = t_cap - f->t_cap_last;
	if (f->t_cap_last > 0 && dt > 0.015 && dt < 0.1) f->period = f->period > 0 ? 0.9 * f->period + 0.1 * dt : dt;
	f->t_cap_prev = f->t_cap_last;
	f->t_cap_last = t_cap;
	f->seq++;
	f->t_arr[f->seq % 4] = now;
	f->phase = g_fg ? 0 : 2;
	f->t_switch = now + (f->period > 0 ? f->period : 1.0 / 30.0); /* no field within a period: show cur */
}

static GLuint g_q_fg;
static void fg_generate(GLuint older, GLuint newer, GLuint mid, GLuint out, float sign)
{
	glBindTextureUnit(13, older);
	glBindTextureUnit(14, newer);
	glBindTextureUnit(15, mid);
	glBindImageTexture(1, out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
	glProgramUniform1f(p_fg, 0, sign);
	glUseProgram(p_fg);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT);
}

/* the engine's field for the push pair seq: generate when it joins exactly prev and cur */
static void fg_field(fgcam *f, const uint32_t seq[2])
{
	if (f->gen && g_fg && seq[1] != f->seq) f->late++;
	if (!f->gen || !g_fg || f->phase != 0 || seq[1] != f->seq || seq[0] + 1 != seq[1]) return;
	glTextureSubImage2D(f->mid, 0, 0, 0, 80, 60, GL_RG, GL_FLOAT, g_rawbuf);
	fg_generate(f->prev, f->cur, f->mid, f->gen, 1.0f);
	f->phase = 1;
	f->made++;
}

/* the texture to show at this refresh. Rules, applied at the draw so no image is replaced unseen: frame
 * seq-1 must have reached the screen before the generated one (a field can arrive in the same loop pass as
 * its frame); the generated image's half period starts at its first refresh. Counts each distinct image
 * once, when it first reaches the screen. */
static GLuint fg_display(fgcam *f, double now)
{
	const uint64_t prev_id = 2ull * (f->seq - 1), gen_id = prev_id + 1;
	const double half = 0.5 * (f->period > 0 ? f->period : 1.0 / 30.0);
	int show = f->phase;
	if (show == 0 && now >= f->t_switch) { f->nofield++; show = f->phase = 2; }
	if (show == 1) {
		if (f->last_id < prev_id) show = 0;                       /* frame seq-1 first, one refresh */
		else if (f->last_id != gen_id) f->t_switch = now + half;  /* first refresh of the generated */
		else if (now + 0.002 >= f->t_switch) show = f->phase = 2; /* 2 ms: a half period of exactly 2 refreshes must not hang on vsync jitter */
	}
	f->show = show;
	const uint64_t id = show == 0 ? prev_id : show == 1 ? gen_id : prev_id + 2;
	if (id != f->last_id) {
		f->last_id = id;
		f->shown++;
		if (!(id & 1) && id >= 2 && id / 2 + 3 >= f->seq && f->t_arr[(id / 2) % 4] > 0) { /* a real frame on screen for the first time */
			const double l = now - f->t_arr[(id / 2) % 4];
			f->lat_sum += l;
			f->lat_n++;
			if (l > f->lat_max) f->lat_max = l;
		}
	}
	return show == 0 ? f->prev : show == 1 ? f->gen : f->cur;
}

/* ==== frame generation test (--fg-test S): skip-one ground truth ===================================
 * A third engine sees only every second Kinect frame. Its field between frames k-1 and k+1 generates the
 * instant of the real frame k, which the test keeps: error against the truth, for four predictions of k:
 * repeat k-1, average of k-1 and k+1, generated, generated with the field's sign reversed (a control: a
 * correct field must beat it). Over all pixels and over the moving ones (k-1 and k+1 differ by > 24 in a
 * channel). Twice the motion of the live case: a stricter test than the doubling itself. */
#define FGT_RING 8
static fsrcam *g_fst;
static GLuint t_ta, t_tb, t_tg, t_tmid;
static uint8_t *g_tring, *g_tgen[2];
static uint32_t g_tidx[FGT_RING], g_tmap[16], g_tpush;
static double g_tt[FGT_RING], g_tsecs, g_t0_test;
static float g_trawbuf[80 * 60 * 2], g_tfsbuf[80 * 60 * 4];
static struct {
	double sse[4][2], ns_gpu;
	unsigned long n[2];
	unsigned triples, rejected, motion_triples, wins, wins_rev, gpu_n;
} g_ft;

static int fgt_setup(void)
{
	static const int ct[4] = {8, 9, 10, 11};
	g_fst = fsr_open(W, H, ct, 4);
	g_tring = malloc((size_t)FGT_RING * W * H * 3);
	g_tgen[0] = malloc((size_t)W * H * 3);
	g_tgen[1] = malloc((size_t)W * H * 3);
	if (!g_fst || !g_tring || !g_tgen[0] || !g_tgen[1]) return -1;
	for (int i = 0; i < FGT_RING; i++) g_tidx[i] = UINT32_MAX;
	t_ta = tex2d(GL_RGB8, W, H, 1);
	t_tb = tex2d(GL_RGB8, W, H, 1);
	t_tg = tex2d(GL_RGBA8, W, H, 1);
	t_tmid = tex2d(GL_RG32F, 80, 60, 1);
	glCreateQueries(GL_TIME_ELAPSED, 1, &g_q_fg);
	return 0;
}

static void fgt_frame(const uint8_t *rgb, uint32_t k, double t_cap)
{
	memcpy(g_tring + (size_t)(k % FGT_RING) * W * H * 3, rgb, (size_t)W * H * 3);
	g_tidx[k % FGT_RING] = k;
	g_tt[k % FGT_RING] = t_cap;
	if (k & 1) return;
	fsr_push(g_fst, rgb);
	g_tmap[++g_tpush % 16] = k;
}

static void fgt_poll(void)
{
	uint32_t seq[2];
	if (!fsr_poll(g_fst, g_tfsbuf, g_trawbuf, seq)) return;
	const uint32_t ka = g_tmap[seq[0] % 16], kb = g_tmap[seq[1] % 16], km = ka + 1;
	if (seq[0] + 1 != seq[1] || kb != ka + 2 || g_tidx[ka % FGT_RING] != ka || g_tidx[km % FGT_RING] != km || g_tidx[kb % FGT_RING] != kb) { g_ft.rejected++; return; }
	const double g1 = g_tt[km % FGT_RING] - g_tt[ka % FGT_RING], g2 = g_tt[kb % FGT_RING] - g_tt[km % FGT_RING];
	if (g1 < 0.025 || g1 > 0.042 || g2 < 0.025 || g2 > 0.042 || fabs(g1 - g2) > 0.006) { g_ft.rejected++; return; } /* k not half way */
	const uint8_t *A = g_tring + (size_t)(ka % FGT_RING) * W * H * 3, *M = g_tring + (size_t)(km % FGT_RING) * W * H * 3,
		      *B = g_tring + (size_t)(kb % FGT_RING) * W * H * 3;
	glTextureSubImage2D(t_ta, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, A);
	glTextureSubImage2D(t_tb, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, B);
	glTextureSubImage2D(t_tmid, 0, 0, 0, 80, 60, GL_RG, GL_FLOAT, g_trawbuf);
	for (int s = 0; s < 2; s++) {
		if (!s) glBeginQuery(GL_TIME_ELAPSED, g_q_fg);
		fg_generate(t_ta, t_tb, t_tmid, t_tg, s ? -1.0f : 1.0f);
		if (!s) {
			glEndQuery(GL_TIME_ELAPSED);
			GLuint64 ns = 0;
			glGetQueryObjectui64v(g_q_fg, GL_QUERY_RESULT, &ns);
			g_ft.ns_gpu += (double)ns;
			g_ft.gpu_n++;
		}
		glGetTextureImage(t_tg, 0, GL_RGB, GL_UNSIGNED_BYTE, W * H * 3, g_tgen[s]);
	}
	double sse[4][2] = {{0}};
	unsigned long nm = 0;
	for (int y = 8; y < H - 8; y++)
		for (int x = 8; x < W - 8; x++) {
			const size_t o = ((size_t)y * W + x) * 3;
			int mot = 0;
			for (int c = 0; c < 3; c++) mot |= abs((int)A[o + c] - (int)B[o + c]) > 24;
			for (int c = 0; c < 3; c++) {
				const int truth = M[o + c], p[4] = {A[o + c], (A[o + c] + B[o + c] + 1) >> 1, g_tgen[0][o + c], g_tgen[1][o + c]};
				for (int q = 0; q < 4; q++) {
					const double d = (double)(p[q] - truth);
					sse[q][0] += d * d;
					if (mot) sse[q][1] += d * d;
				}
			}
			nm += (unsigned long)mot;
		}
	for (int q = 0; q < 4; q++) { g_ft.sse[q][0] += sse[q][0]; g_ft.sse[q][1] += sse[q][1]; }
	g_ft.n[0] += 3ul * (unsigned long)(W - 16) * (H - 16);
	g_ft.n[1] += 3ul * nm;
	g_ft.triples++;
	if (nm >= 1000) {
		g_ft.motion_triples++;
		g_ft.wins += sse[2][1] < sse[1][1];
		g_ft.wins_rev += sse[2][1] < sse[3][1];
	}
}

static double fgt_psnr(int q, int m) { return g_ft.sse[q][m] > 0 ? 10.0 * log10(255.0 * 255.0 * (double)g_ft.n[m] / g_ft.sse[q][m]) : 99.0; }

static void fgt_report(void)
{
	static const char *const NAME[4] = {"repeat k-1", "average", "generated", "sign reversed"};
	printf("fg-test: %u triples measured, %u rejected (uneven timing or a skipped push), %u with >= 1000 moving pixels, "
	       "moving pixels %.2f %% of all\n", g_ft.triples, g_ft.rejected, g_ft.motion_triples,
	       g_ft.n[0] ? 100.0 * (double)g_ft.n[1] / (double)g_ft.n[0] : 0.0);
	for (int q = 0; q < 4; q++) printf("fg-test: %-14s PSNR all %6.2f dB   moving %6.2f dB\n", NAME[q], fgt_psnr(q, 0), fgt_psnr(q, 1));
	printf("fg-test: generated beats average in %u of %u moving triples, beats sign reversed in %u; generation GPU %.4f ms (mean of %u)\n",
	       g_ft.wins, g_ft.motion_triples, g_ft.wins_rev, g_ft.gpu_n ? g_ft.ns_gpu * 1e-6 / g_ft.gpu_n : 0.0, g_ft.gpu_n);
}

/* ==== flow lock (host) ========================================================================== */

static void lock_setup(void)
{
	GLuint rb;
	static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	t_sgray = tex2d(GL_R32F, W, H, 0);
	t_sX = tex2d(GL_RGBA32F, W, H, 0);
	t_synf = tex2d(GL_R32F, W, H, 0);
	glCreateRenderbuffers(1, &rb);
	glNamedRenderbufferStorage(rb, GL_DEPTH_COMPONENT24, W, H);
	glCreateFramebuffers(1, &fbo_syn);
	glNamedFramebufferTexture(fbo_syn, GL_COLOR_ATTACHMENT0, t_sgray, 0);
	glNamedFramebufferTexture(fbo_syn, GL_COLOR_ATTACHMENT1, t_sX, 0);
	glNamedFramebufferRenderbuffer(fbo_syn, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
	glNamedFramebufferDrawBuffers(fbo_syn, 2, bufs);
	if (glCheckNamedFramebufferStatus(fbo_syn, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		fprintf(stderr, "kinect3d: lock framebuffer incomplete\n");
	glCreateBuffers(1, &ssbo_part);
	glNamedBufferStorage(ssbo_part, sizeof(float) * 29 * (W / 16) * (H / 16), NULL, GL_DYNAMIC_STORAGE_BIT);
	g_flock = flow_create(W, H, "lock");
}

/* GL vertex (X, -Y, -Z) -> Kinect frame -> webcam frame (R, t) -> webcam pixels (pixel centres at
 * integers, row 0 = top, as the webcam texture) -> clip space */
static mat4 webcam_mvp(void)
{
	const float *R = g_cal.R, *t = g_cal.t, n = 50.0f, f = 30000.0f;
	const float P[4][4] = {
		{2.0f * g_cal.K[0] / W, 0.0f, 2.0f * (g_cal.K[2] + 0.5f) / W - 1.0f, 0.0f},
		{0.0f, 2.0f * g_cal.K[1] / H, 2.0f * (g_cal.K[3] + 0.5f) / H - 1.0f, 0.0f},
		{0.0f, 0.0f, (f + n) / (f - n), -2.0f * f * n / (f - n)},
		{0.0f, 0.0f, 1.0f, 0.0f},
	};
	const float E[4][4] = {
		{R[0], -R[1], -R[2], t[0]}, {R[3], -R[4], -R[5], t[1]}, {R[6], -R[7], -R[8], t[2]}, {0.0f, 0.0f, 0.0f, 1.0f},
	};
	mat4 m;
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++) {
			float s = 0;
			for (int k = 0; k < 4; k++) s += P[r][k] * E[k][c];
			m.m[c * 4 + r] = s;
		}
	return m;
}

static int chol6(double A[6][6], double b[6], double x[6])
{
	double L[6][6] = {{0}}, y[6];
	for (int i = 0; i < 6; i++)
		for (int j = 0; j <= i; j++) {
			double s = A[i][j];
			for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
			if (i == j) { if (s <= 0) return -1; L[i][i] = sqrt(s); }
			else L[i][j] = s / L[j][j];
		}
	for (int i = 0; i < 6; i++) { double s = b[i]; for (int k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
	for (int i = 5; i >= 0; i--) { double s = y[i]; for (int k = i + 1; k < 6; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
	return 0;
}


/* coarse acquisition: global NCC shift between the synthetic view (valid pixels only) and the real
 * webcam gray (measured 14:12: the face calibration left the pose 43 px off, beyond the flow's reach);
 * a shift (dx, dy) is applied as the camera rotation w = (-dy/f, dx/f, 0): R' = exp(w) R, t' = exp(w) t */
static float g_syn[NPIX], g_real[NPIX];

static double shift_ncc(int dx, int dy, int stride)
{
	double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
	unsigned n = 0;
	for (int y = 0; y < H; y += stride)
		for (int x = 0; x < W; x += stride) {
			const float a = g_syn[y * W + x];
			const int u = x + dx, v = y + dy;
			if (a < 0 || u < 0 || v < 0 || u >= W || v >= H) continue;
			const double b = g_real[v * W + u];
			sa += a; sb += b; saa += (double)a * a; sbb += b * b; sab += a * b; n++;
		}
	if (n < 300) return -2;
	const double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
	return va > 0 && vb > 0 ? cov / sqrt(va * vb) : -2;
}

static void rotate_pose(double wx, double wy, double wz)
{
	const double wn = sqrt(wx * wx + wy * wy + wz * wz);
	if (wn <= 0) return;
	const double kx = wx / wn, ky = wy / wn, kz = wz / wn, sn = sin(wn), cs = 1 - cos(wn);
	const double K[3][3] = {{0, -kz, ky}, {kz, 0, -kx}, {-ky, kx, 0}};
	double Rw[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			double k2 = 0;
			for (int m = 0; m < 3; m++) k2 += K[i][m] * K[m][j];
			Rw[i][j] = (i == j) + sn * K[i][j] + cs * k2;
		}
	float R2[9], t2[3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) R2[i * 3 + j] = (float)(Rw[i][0] * g_cal.R[j] + Rw[i][1] * g_cal.R[3 + j] + Rw[i][2] * g_cal.R[6 + j]);
		t2[i] = (float)(Rw[i][0] * g_cal.t[0] + Rw[i][1] * g_cal.t[1] + Rw[i][2] * g_cal.t[2]);
	}
	memcpy(g_cal.R, R2, sizeof R2);
	memcpy(g_cal.t, t2, sizeof t2);
	apply_calib();
}

static void lock_coarse(void)
{
	glFinish();
	glGetTextureImage(t_sgray, 0, GL_RED, GL_FLOAT, sizeof g_syn, g_syn);
	glGetTextureImage(flow_gray(g_fw, 1), 0, GL_RED, GL_FLOAT, sizeof g_real, g_real);
	double best = -3, c0 = shift_ncc(0, 0, 2);
	int bx = 0, by = 0;
	for (int dy = -120; dy <= 120; dy += 4)
		for (int dx = -120; dx <= 120; dx += 4) {
			const double c = shift_ncc(dx, dy, 4);
			if (c > best) { best = c; bx = dx; by = dy; }
		}
	const int cx = bx, cy = by;
	best = -3;
	for (int dy = cy - 4; dy <= cy + 4; dy++)
		for (int dx = cx - 4; dx <= cx + 4; dx++) {
			const double c = shift_ncc(dx, dy, 2);
			if (c > best) { best = c; bx = dx; by = dy; }
		}
	double second = -3; /* sharpness: best candidate more than 12 px from the peak (stride-4 grid) */
	for (int dy = -120; dy <= 120; dy += 4)
		for (int dx = -120; dx <= 120; dx += 4)
			if (abs(dx - bx) > 12 || abs(dy - by) > 12) { const double c = shift_ncc(dx, dy, 4); if (c > second) second = c; }
	const int sharp = best - second >= 0.02 && abs(bx) <= 80 && abs(by) <= 80;
	printf("lock: coarse shift (%+d, %+d) px, NCC %.4f (at 0,0: %.4f, runner-up %.4f): %s\n", bx, by, best, c0, second,
	       sharp ? "applied" : "ambiguous, pose kept");
	if (!sharp) return;
	if (best > c0 + 0.01) rotate_pose(-(double)by / g_cal.K[1], (double)bx / g_cal.K[0], 0.0);
}
static void lock_step(void)
{
	static const float neg1[4] = {-1, 0, 0, 0}, zero[4] = {0, 0, 0, 0}, one = 1.0f;
	static float part[29 * (W / 16) * (H / 16)];
	mat4 mvp = webcam_mvp();
	/* 1. the Kinect mesh as the webcam would see it */
	glBindFramebuffer(GL_FRAMEBUFFER, fbo_syn);
	glViewport(0, 0, W, H);
	glClearNamedFramebufferfv(fbo_syn, GL_COLOR, 0, neg1);
	glClearNamedFramebufferfv(fbo_syn, GL_COLOR, 1, zero);
	glClearNamedFramebufferfv(fbo_syn, GL_DEPTH, 0, &one);
	glEnable(GL_DEPTH_TEST);
	glProgramUniformMatrix4fv(p_syn, glGetUniformLocation(p_syn, "u_mvp"), 1, GL_FALSE, mvp.m);
	glProgramUniform1f(p_syn, glGetUniformLocation(p_syn, "u_cull"), g_cull);
	glProgramUniform1i(p_syn, glGetUniformLocation(p_syn, "u_occl"), 1);
	glUseProgram(p_syn);
	glBindVertexArray(vao[0]);
	glDrawElements(GL_TRIANGLES, NVERT, GL_UNSIGNED_INT, NULL);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
	/* 2. holes take the real webcam gray */
	glBindTextureUnit(14, t_sgray);
	glBindTextureUnit(15, flow_gray(g_fw, 1));
	glBindImageTexture(2, t_synf, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
	glUseProgram(p_comp);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
	/* 3. synthetic -> real flow */
	flow_pair(g_flock, t_synf, flow_gray(g_fw, 1));
	/* 4. Gauss-Newton sums */
	glBindTextureUnit(14, flow_result(g_flock));
	glBindTextureUnit(15, t_sX);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_part);
	glUseProgram(p_gn);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	glGetNamedBufferSubData(ssbo_part, 0, sizeof part, part);
	double s[29] = {0};
	for (int g = 0; g < (W / 16) * (H / 16); g++)
		for (int k = 0; k < 29; k++) s[k] += part[g * 29 + k];
	g_lk.n = (unsigned)s[28];
	g_lk.mean_r = s[28] > 0 ? s[27] / s[28] : 0;
	static int starved = 1000;
	starved = s[28] < 2000 ? starved + 1 : 0;
	if (g_lock && starved >= 30) { lock_coarse(); starved = 0; } /* acquire on start, re-acquire when lost */
	if (!g_lock || s[28] < 2000) return;
	double A[6][6], b[6], x[6];
	for (int i = 0, k = 0; i < 6; i++)
		for (int j = i; j < 6; j++, k++) A[i][j] = A[j][i] = s[k];
	for (int i = 0; i < 6; i++) { A[i][i] *= 1.001; b[i] = s[21 + i]; }
	if (chol6(A, b, x)) return;
	for (int i = 0; i < 6; i++) x[i] *= 0.5; /* damped */
	double wn = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), tn = sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
	const double wmax = 0.5 * M_PI / 180.0, tmax = 5.0;
	if (wn > wmax) { for (int i = 0; i < 3; i++) x[i] *= wmax / wn; wn = wmax; }
	if (tn > tmax) { for (int i = 3; i < 6; i++) x[i] *= tmax / tn; tn = tmax; }
	/* R' = exp(w) R, t' = exp(w) t + dt (Rodrigues) */
	double Rw[3][3], kx = 0, ky = 0, kz = 0, sn = sin(wn), cs = 1 - cos(wn);
	if (wn > 0) { kx = x[0] / wn; ky = x[1] / wn; kz = x[2] / wn; }
	const double K[3][3] = {{0, -kz, ky}, {kz, 0, -kx}, {-ky, kx, 0}};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			double k2 = 0;
			for (int m = 0; m < 3; m++) k2 += K[i][m] * K[m][j];
			Rw[i][j] = (i == j) + sn * K[i][j] + cs * k2;
		}
	float R2[9], t2[3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			R2[i * 3 + j] = (float)(Rw[i][0] * g_cal.R[j] + Rw[i][1] * g_cal.R[3 + j] + Rw[i][2] * g_cal.R[6 + j]);
		t2[i] = (float)(Rw[i][0] * g_cal.t[0] + Rw[i][1] * g_cal.t[1] + Rw[i][2] * g_cal.t[2] + x[3 + i]);
	}
	memcpy(g_cal.R, R2, sizeof R2);
	memcpy(g_cal.t, t2, sizeof t2);
	apply_calib();
	g_lk.step_deg = wn * 180.0 / M_PI;
	g_lk.step_mm = tn;
	g_lk.updates++;
}


/* --dump-lock: synthetic (Kinect mesh seen from the webcam pose) and real webcam gray, red/cyan anaglyph
 * (synthetic red, real cyan: misalignment shows as coloured fringes), and the lock flow's valid mask */
static void dump_lock(const char *dir)
{
	float *a = malloc(sizeof(float) * NPIX), *b = malloc(sizeof(float) * NPIX), *f = malloc(sizeof(float) * NPIX * 4);
	uint8_t *rgb = malloc(NPIX * 3);
	char path[1024];
	if (!a || !b || !f || !rgb) return;
	glFinish();
	glGetTextureImage(t_synf, 0, GL_RED, GL_FLOAT, sizeof(float) * NPIX, a);
	glGetTextureImage(flow_gray(g_fw, 1), 0, GL_RED, GL_FLOAT, sizeof(float) * NPIX, b);
	glGetTextureImage(flow_result(g_flock), 0, GL_RGBA, GL_FLOAT, sizeof(float) * NPIX * 4, f);
	unsigned valid = 0;
	for (int i = 0; i < NPIX; i++) {
		uint8_t sa = (uint8_t)fminf(fmaxf(a[i], 0), 255), sb = (uint8_t)fminf(fmaxf(b[i], 0), 255);
		rgb[3 * i] = sa;
		rgb[3 * i + 1] = rgb[3 * i + 2] = sb;
		valid += f[4 * i + 3] > 0.5f;
	}
	snprintf(path, sizeof path, "%s/lock-anaglyph.ppm", dir);
	FILE *o = fopen(path, "wb");
	if (o) { fprintf(o, "P6\n%d %d\n255\n", W, H); fwrite(rgb, 3, NPIX, o); fclose(o); }
	for (int i = 0; i < NPIX; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = f[4 * i + 3] > 0.5f ? 255 : 0;
	snprintf(path, sizeof path, "%s/lock-valid.ppm", dir);
	o = fopen(path, "wb");
	if (o) { fprintf(o, "P6\n%d %d\n255\n", W, H); fwrite(rgb, 3, NPIX, o); fclose(o); }
	{ /* raw synthetic gray (-1 = no Kinect colour) and real gray for the shift search */
		glGetTextureImage(t_sgray, 0, GL_RED, GL_FLOAT, sizeof(float) * NPIX, a);
		snprintf(path, sizeof path, "%s/lock-syn.f32", dir);
		o = fopen(path, "wb");
		if (o) { fwrite(a, sizeof(float), NPIX, o); fclose(o); }
		snprintf(path, sizeof path, "%s/lock-real.f32", dir);
		o = fopen(path, "wb");
		if (o) { fwrite(b, sizeof(float), NPIX, o); fclose(o); }
	}
	printf("dump-lock: %u valid flow pixels; anaglyph and mask in %s\n", valid, dir);
	free(a); free(b); free(f); free(rgb);
}
static int save_calib(const char *path)
{
	FILE *o = fopen(path, "w");
	if (!o) return -1;
	time_t now = time(NULL);
	char stamp[64];
	strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", localtime(&now));
	fprintf(o, "# kinect3d webcam calibration, pose locked by flow at %s (%u updates, mean residual %.3f px on %u px)\n", stamp,
		g_lk.updates, g_lk.mean_r, g_lk.n);
	fprintf(o, "K %.6f %.6f %.6f %.6f\nD %.8f %.8f\nR", (double)g_cal.K[0], (double)g_cal.K[1], (double)g_cal.K[2], (double)g_cal.K[3],
		(double)g_cal.D[0], (double)g_cal.D[1]);
	for (int i = 0; i < 9; i++) fprintf(o, " %.9f", (double)g_cal.R[i]);
	fprintf(o, "\nt %.4f %.4f %.4f\n", (double)g_cal.t[0], (double)g_cal.t[1], (double)g_cal.t[2]);
	fprintf(o, "gain %.5f %.5f %.5f\noffset %.5f %.5f %.5f\n", (double)g_cal.gain[0], (double)g_cal.gain[1], (double)g_cal.gain[2],
		(double)g_cal.off[0], (double)g_cal.off[1], (double)g_cal.off[2]);
	return fclose(o);
}
/* --calib-dump: Kinect RGB, webcam RGB (PPM) and the RGB-space point map (RGBA32F raw) of this moment */
static void compute_pmap(void)
{
	static const float zero[4] = {0, 0, 0, 0};
	glClearTexImage(t_pmap, 0, GL_RGBA, GL_FLOAT, zero);
	glProgramUniform1f(p_pmap, glGetUniformLocation(p_pmap, "u_k"), g_k);
	glBindImageTexture(4, t_pmap, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glUseProgram(p_pmap);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

static int dump_calib(const char *dir, int k)
{
	uint8_t *img = malloc(NPIX * 3);
	float *pm = malloc(sizeof(float) * NPIX * 4);
	char path[1024];
	int err = 0;
	if (!img || !pm) return 1;
	compute_pmap();
	const GLuint src[2] = {t_rgb, t_wrgb};
	const char *name[2] = {"kinect", "webcam"};
	for (int c = 0; c < 2; c++) {
		glGetTextureImage(src[c], 0, GL_RGB, GL_UNSIGNED_BYTE, NPIX * 3, img);
		snprintf(path, sizeof path, "%s/snap%d-%s.ppm", dir, k, name[c]);
		FILE *o = fopen(path, "wb");
		err |= !o || fprintf(o, "P6\n%d %d\n255\n", W, H) < 0 || fwrite(img, 3, NPIX, o) != NPIX || fclose(o);
	}
	glGetTextureImage(t_pmap, 0, GL_RGBA, GL_FLOAT, sizeof(float) * NPIX * 4, pm);
	unsigned n = 0;
	for (int i = 0; i < NPIX; i++) n += pm[4 * i + 3] > 0.5f;
	snprintf(path, sizeof path, "%s/snap%d-points.f32", dir, k);
	FILE *o = fopen(path, "wb");
	err |= !o || fwrite(pm, sizeof(float) * 4, NPIX, o) != NPIX || fclose(o);
	printf("calib-dump %d: %u RGB pixels with a 3D point (%.1f %%)%s\n", k, n, 100.0 * n / NPIX, err ? ", WRITE ERROR" : "");
	free(img);
	free(pm);
	return err;
}

/* ==== --calib-track: one bright moving light seen by both cameras =============================== */
enum { BW = W / 8, BH = H / 8 };
typedef struct { uint16_t cnt[BW * BH]; uint32_t seen[BW * BH]; uint8_t stat[BW * BH]; uint32_t frames; } brightmask;

/* after learning: a block is static if it or a neighbour was near-white in >= 30 % of the frames */
static void bm_finish(brightmask *m)
{
	for (int by = 0; by < BH; by++)
		for (int bx = 0; bx < BW; bx++) {
			uint8_t s = 0;
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++) {
					int x = bx + dx, y = by + dy;
					if (x >= 0 && y >= 0 && x < BW && y < BH) s |= m->cnt[y * BW + x] * 10u >= m->frames * 3u;
				}
			m->stat[by * BW + bx] = s;
		}
}

/* learn: count near-white blocks. detect: centroid of near-white pixels outside static blocks, kept only
 * when they form one compact blob (3..3000 px inside an 80 px box) */
static int bright_blob(const uint8_t *rgb, brightmask *m, int learn, float *cx, float *cy, unsigned *npx)
{
	long sx = 0, sy = 0;
	unsigned c = 0;
	int x0 = W, y0 = H, x1 = -1, y1 = -1;
	m->frames += (uint32_t)learn;
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			const uint8_t *p = rgb + 3 * (y * W + x);
			uint8_t mn = p[0] < p[1] ? p[0] : p[1];
			mn = mn < p[2] ? mn : p[2];
			if (mn < 235) continue;
			int b = (y >> 3) * BW + (x >> 3);
			if (learn) {
				if (m->seen[b] != m->frames) { m->seen[b] = m->frames; m->cnt[b]++; }
				continue;
			}
			if (m->stat[b]) continue;
			sx += x; sy += y; c++;
			if (x < x0) x0 = x;
			if (x > x1) x1 = x;
			if (y < y0) y0 = y;
			if (y > y1) y1 = y;
		}
	if (learn || c < 3 || c > 3000 || x1 - x0 > 80 || y1 - y0 > 80) return 0;
	*cx = (float)sx / (float)c;
	*cy = (float)sy / (float)c;
	*npx = c;
	return 1;
}



/* --analyze: why triangles vanish. Same triangles and test as the mesh (keep3), on one live depth frame.
 * Edge cuts are compared with the Kinect's depth step at that range, q(z) ~ z^2 / 348000 mm
 * (f 580 px, baseline 75 mm, 1/8 px disparity: arithmetic), to tell false cuts from silhouettes. */
static void analyze_depth(const uint16_t *d)
{
	enum { NB = 6 };
	static const int edge_mm[NB + 1] = {0, 1000, 1500, 2000, 3000, 4000, 10000};
	unsigned zero = 0, far = 0, tri[NB] = {0}, hole[NB] = {0}, cut[NB] = {0}, cut_small[NB] = {0}, keep[NB] = {0};
	for (int i = 0; i < NPIX; i++) { zero += d[i] == 0; far += d[i] >= ZMAX; }
	static const uint8_t pick[6] = {0, 1, 2, 1, 3, 2};
	for (int y = 0; y < H - 1; y++)
		for (int x = 0; x < W - 1; x++) {
			const unsigned c4[4] = {d[y * W + x], d[y * W + x + 1], d[(y + 1) * W + x], d[(y + 1) * W + x + 1]};
			for (int t = 0; t < 2; t++) {
				unsigned a = c4[pick[3 * t]], b = c4[pick[3 * t + 1]], c = c4[pick[3 * t + 2]];
				unsigned mn = a < b ? a : b, mx = a > b ? a : b;
				mn = c < mn ? c : mn;
				mx = c > mx ? c : mx;
				unsigned nz = (a ? a : 99999), m2 = b && b < nz ? b : nz; /* nearest valid corner for banding */
				m2 = c && c < m2 ? c : m2;
				int band = 0;
				while (band < NB - 1 && m2 >= (unsigned)edge_mm[band + 1]) band++;
				tri[band]++;
				if (mn == 0 || mx >= ZMAX) { hole[band]++; continue; }
				if ((float)(mx - mn) <= g_cull * (float)mn) { keep[band]++; continue; }
				cut[band]++;
				if ((double)(mx - mn) <= 3.0 * (double)mn * (double)mn / 348000.0) cut_small[band]++;
			}
		}
	printf("analyze: %u of %d pixels without depth (%.1f %%), %u at or beyond 10 m; cull %.1f %% of the nearest depth\n", zero, NPIX,
	       100.0 * zero / NPIX, far, g_cull * 100.0);
	printf("analyze: depth band      triangles   kept %%   missing corner %%   edge cut %%   of which within 3 depth steps %%\n");
	for (int b = 0; b < NB; b++) {
		if (!tri[b]) continue;
		printf("analyze: %5d-%5d mm %10u %8.1f %18.1f %12.1f %22.1f\n", edge_mm[b], edge_mm[b + 1], tri[b], 100.0 * keep[b] / tri[b],
		       100.0 * hole[b] / tri[b], 100.0 * cut[b] / tri[b], cut[b] ? 100.0 * cut_small[b] / cut[b] : 0.0);
	}
	/* zero runs along rows: a regular hole pattern shows as one dominant run length */
	unsigned runs[17] = {0};
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W;) {
			if (d[y * W + x]) { x++; continue; }
			int r = 0;
			while (x < W && !d[y * W + x]) { r++; x++; }
			runs[r > 16 ? 16 : r]++;
		}
	/* what borders the 8-pixel runs: relative depth step between the valid pixels left and right */
	{
		unsigned bins[6] = {0}, vert_ok = 0, n8 = 0;
		for (int y = 1; y < H - 1; y++)
			for (int x = 1; x < W - 9; x++) {
				if (d[y * W + x - 1] == 0 || d[y * W + x] != 0) continue;
				int r = 0;
				while (x + r < W && d[y * W + x + r] == 0) r++;
				if (r != 8 || x + 8 >= W || d[y * W + x + 8] == 0) { x += r; continue; }
				double zl = d[y * W + x - 1], zr = d[y * W + x + 8], rel = fabs(zl - zr) / fmin(zl, zr);
				int b = rel < 0.005 ? 0 : rel < 0.01 ? 1 : rel < 0.03 ? 2 : rel < 0.1 ? 3 : rel < 0.3 ? 4 : 5;
				bins[b]++;
				n8++;
				vert_ok += d[(y - 1) * W + x + 4] != 0 && d[(y + 1) * W + x + 4] != 0;
				x += r;
			}
		printf("analyze: 8-runs bounded on both sides: %u; left-right depth step <0.5 %% %u, 0.5-1 %% %u, 1-3 %% %u, 3-10 %% %u, 10-30 %% %u, >30 %% %u; "
		       "valid above and below the middle: %u\n", n8, bins[0], bins[1], bins[2], bins[3], bins[4], bins[5], vert_ok);
	}
	/* 8-run geometry: start column mod 8 (a tile grid shows as one dominant phase) and run height */
	{
		unsigned phase[8] = {0}, n = 0, tall = 0;
		for (int y = 0; y < H; y++)
			for (int x = 0; x < W;) {
				if (d[y * W + x]) { x++; continue; }
				int r = 0;
				while (x + r < W && !d[y * W + x + r]) r++;
				if (r == 8) {
					phase[x & 7]++;
					n++;
					int h = 1;
					while (y + h < H && !d[(y + h) * W + x] && !d[(y + h) * W + x + 7]) h++;
					tall += (unsigned)h;
				}
				x += r;
			}
		printf("analyze: 8-runs %u, start column mod 8:", n);
		for (int k = 0; k < 8; k++) printf(" %u", phase[k]);
		printf("; mean height of the hole below each run start %.1f rows\n", n ? (double)tall / n : 0.0);
	}
	{ /* 8-bit view for the eye: near bright, far dark, no depth black; written once per call */
		static int call;
		char path[256];
		uint8_t *v = malloc(NPIX);
		if (v) {
			for (int i = 0; i < NPIX; i++) v[i] = d[i] && d[i] < ZMAX ? (uint8_t)(255 - (d[i] > 5000 ? 5000 : d[i]) * 230 / 5000) : 0;
			snprintf(path, sizeof path, "out/depth-%s.pgm", call++ ? "filtered" : "raw");
			FILE *o = fopen(path, "wb");
			if (o) { fprintf(o, "P5\n%d %d\n255\n", W, H); fwrite(v, 1, NPIX, o); fclose(o); }
			free(v);
		}
	}
	printf("analyze: zero runs along rows, length 1..15, 16+:");
	for (int r = 1; r <= 16; r++) printf(" %u", runs[r]);
	printf("\n");
}
/* 3D points (Kinect frame, mm) of the 5 face landmarks seen in the Kinect RGB: one point-map read of the
 * landmarks' box + 3 px, nearest valid point within 3 px of each landmark; NaN when none. Returns count. */
static int face_3d(const face5 *f, float X[15])
{
	float x0 = f->lm[0], x1 = f->lm[0], y0 = f->lm[1], y1 = f->lm[1];
	for (int l = 1; l < 5; l++) {
		x0 = fminf(x0, f->lm[2 * l]); x1 = fmaxf(x1, f->lm[2 * l]);
		y0 = fminf(y0, f->lm[2 * l + 1]); y1 = fmaxf(y1, f->lm[2 * l + 1]);
	}
	int bx = (int)x0 - 4, by = (int)y0 - 4, bw = (int)x1 - bx + 5, bh = (int)y1 - by + 5;
	bx = bx < 0 ? 0 : bx;
	by = by < 0 ? 0 : by;
	if (bx + bw > W) bw = W - bx;
	if (by + bh > H) bh = H - by;
	int n = 0;
	for (int k = 0; k < 15; k++) X[k] = NAN;
	if (bw <= 0 || bh <= 0 || bw > 320 || bh > 320) return 0;
	float *buf = malloc(sizeof(float) * 4 * (size_t)bw * (size_t)bh);
	if (!buf) return 0;
	glGetTextureSubImage(t_pmap, 0, bx, by, 0, bw, bh, 1, GL_RGBA, GL_FLOAT, (GLsizei)(sizeof(float) * 4 * (size_t)bw * (size_t)bh), buf);
	for (int l = 0; l < 5; l++) {
		int cx = (int)lrintf(f->lm[2 * l]) - bx, cy = (int)lrintf(f->lm[2 * l + 1]) - by, best = 99;
		for (int dy = -3; dy <= 3; dy++)
			for (int dx = -3; dx <= 3; dx++) {
				int x = cx + dx, y = cy + dy, d2 = dx * dx + dy * dy;
				if (x < 0 || y < 0 || x >= bw || y >= bh || d2 >= best) continue;
				const float *p = buf + 4 * ((size_t)y * bw + x);
				if (p[3] > 0.5f) { best = d2; X[3 * l] = p[0]; X[3 * l + 1] = p[1]; X[3 * l + 2] = p[2]; }
			}
		n += best < 99;
	}
	free(buf);
	return n;
}
/* 3D point (Kinect frame, mm) at a Kinect RGB pixel: points of the newest point map within 10 px whose
 * depth is within 30 mm of their median, averaged */
static int kinect_point(float cx, float cy, float out[3])
{
	enum { S = 21 };
	float buf[S * S * 4], z[S * S];
	int x0 = (int)cx - S / 2, y0 = (int)cy - S / 2, n = 0;
	x0 = x0 < 0 ? 0 : x0 > W - S ? W - S : x0;
	y0 = y0 < 0 ? 0 : y0 > H - S ? H - S : y0;
	glGetTextureSubImage(t_pmap, 0, x0, y0, 0, S, S, 1, GL_RGBA, GL_FLOAT, sizeof buf, buf);
	for (int i = 0; i < S * S; i++)
		if (buf[4 * i + 3] > 0.5f) z[n++] = buf[4 * i + 2];
	if (n < 5) return 0;
	for (int i = 1; i < n; i++) /* insertion sort, at most 441 values */
		for (int j = i; j > 0 && z[j] < z[j - 1]; j--) { float t = z[j]; z[j] = z[j - 1]; z[j - 1] = t; }
	float med = z[n / 2], s[3] = {0, 0, 0};
	int m = 0;
	for (int i = 0; i < S * S; i++)
		if (buf[4 * i + 3] > 0.5f && fabsf(buf[4 * i + 2] - med) < 30.0f) {
			s[0] += buf[4 * i]; s[1] += buf[4 * i + 1]; s[2] += buf[4 * i + 2]; m++;
		}
	for (int k = 0; k < 3; k++) out[k] = s[k] / (float)m;
	return 1;
}

/* the point map (Kinect RGB pixel -> 3D point of the newest depth frame) is computed only in the calibration
 * modes' passes otherwise: this keeps it current for the face, once per depth frame at most */
static void pmap_fresh(void)
{
	if (g_pmap_at == g_depth_up) return;
	compute_pmap();
	g_pmap_at = g_depth_up;
}

/* a new result of the Kinect's face tracker: where the face is in the room, and how fast it moves. Measured
 * frames only (a prediction is not a position). The point map is the newest depth frame's: up to one depth
 * frame (33 ms) apart from the RGB frame the box is of. */
static void face_update(void)
{
	const vtf_out *fo = &g_fo[0];
	if (fo->state != VTF_OBSERVED && fo->state != VTF_NEW) return;
	pmap_fresh();
	float P[3];
	if (!kinect_point(fo->cx, fo->cy, P)) { g_facep.misses++; return; }
	double q[3];
	for (int i = 0; i < 3; i++) q[i] = g_room.Rnow[3 * i] * P[0] + g_room.Rnow[3 * i + 1] * P[1] + g_room.Rnow[3 * i + 2] * P[2];
	const unsigned k = g_facep.k++ % 16;
	g_facep.t[k] = fo->t;
	memcpy(g_facep.p[k], q, sizeof q);
	if (g_facep.n < 16) g_facep.n++;
	g_facep.points++;
	memcpy(g_facep.pos, q, sizeof q);
	g_facep.have = 1;
	g_facep.t_last = fo->t;
	double st = 0, stt = 0, sp[3] = {0, 0, 0}, stp[3] = {0, 0, 0};
	unsigned m = 0;
	for (unsigned j = 0; j < g_facep.n; j++) {
		const double dt = g_facep.t[j] - fo->t;
		if (dt < -0.3 || dt > 0) continue;
		st += dt;
		stt += dt * dt;
		for (int i = 0; i < 3; i++) { sp[i] += g_facep.p[j][i]; stp[i] += dt * g_facep.p[j][i]; }
		m++;
	}
	const double den = (double)m * stt - st * st;
	for (int i = 0; i < 3; i++) g_facep.vel[i] = m >= 4 && den > 1e-9 ? ((double)m * stp[i] - st * sp[i]) / den : 0.0;
}

/* ==== face alignment: the webcam's pose from the face both cameras see (user, 15:08) =====================
 * A pair = one YuNet detection per camera, the two frames within 50 ms of each other, while the face holds still
 * in both (tracker rate < 30 px/s, landmarks within 3 px of that camera's previous detection): the cameras are
 * not synchronised and a tracker result lags its frame. The Kinect's landmarks become 3D points (point map), the
 * webcam's are their pixels: up to 5 correspondences a pair. A pair is kept only when the nose moved 15 mm from the
 * last kept one or 1 s passed, so a still head cannot flood the set.
 * The solve: PnP over R, t (c = R X + t, the occlusion fill's own model), K and k1 k2 fixed (focal 690.75 from the
 * spec), Huber 3 px, Levenberg-damped Gauss-Newton from the pose in use, rotation as R <- exp([w]) R. Judged on
 * time blocks it never saw (alternate 5 s blocks train / test): applied only when its median error there beats
 * the pose in use by 10 %, with >= 100 test correspondences and >= 150 mm of head travel in the set. */
#define AL_MAX 4096
static struct {
	double X[AL_MAX][3], u[AL_MAX][2], t[AL_MAX];
	int n, last_solve_n, have_k, have_w, okk[5];
	double tk, tw, t_acc, nose_acc[3], lo[3], hi[3];
	float Xk[5][3], uw[5][2], lm_prev[2][10], lmw[10], rw[2];
	double t_prev[2];
	uint64_t seq[2];
	unsigned pairs, kept, still_k, still_w, unstill, untimed, solves, applied, resets;
	double old_med, new_med, train_rms, d_rot, d_t, spread, ncc_old, ncc_new;
	int n_test;
} g_al;
static int g_align = 1;

static void al_project(const double R[9], const double t[3], const double X[3], double p[2])
{
	double c[3];
	for (int i = 0; i < 3; i++) c[i] = R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + t[i];
	const double nx = c[0] / c[2], ny = c[1] / c[2], r2 = nx * nx + ny * ny, d = 1.0 + r2 * (g_cal.D[0] + r2 * g_cal.D[1]);
	p[0] = g_cal.K[0] * nx * d + g_cal.K[2];
	p[1] = g_cal.K[1] * ny * d + g_cal.K[3];
}

/* R <- exp([w]) R (Rodrigues) */
static void al_rotate(double R[9], const double w[3])
{
	const double th = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
	double E[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, N[9];
	if (th > 1e-15) {
		const double k[3] = {w[0] / th, w[1] / th, w[2] / th}, s = sin(th), c = 1.0 - cos(th);
		const double Kx[9] = {0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0};
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				double kk = 0;
				for (int m = 0; m < 3; m++) kk += Kx[3 * i + m] * Kx[3 * m + j];
				E[3 * i + j] += s * Kx[3 * i + j] + c * kk;
			}
	}
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) N[3 * i + j] = E[3 * i] * R[j] + E[3 * i + 1] * R[3 + j] + E[3 * i + 2] * R[6 + j];
	memcpy(R, N, sizeof N);
}

static int al_block(int i) { return (int)((long)floor(g_al.t[i] / 5.0) & 1); }

static double al_cost(const double R[9], const double t[3], int part)
{
	double cost = 0, p[2];
	for (int i = 0; i < g_al.n; i++) {
		if (al_block(i) != part) continue;
		al_project(R, t, g_al.X[i], p);
		const double e = hypot(p[0] - g_al.u[i][0], p[1] - g_al.u[i][1]);
		cost += e <= 3.0 ? 0.5 * e * e : 3.0 * (e - 1.5);
	}
	return cost;
}

static int al_cmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* median reprojection error over one part; *cnt correspondences in it */
static double al_median(const double R[9], const double t[3], int part, int *cnt)
{
	static double e[AL_MAX];
	int m = 0;
	double p[2];
	for (int i = 0; i < g_al.n; i++) {
		if (al_block(i) != part) continue;
		al_project(R, t, g_al.X[i], p);
		e[m++] = hypot(p[0] - g_al.u[i][0], p[1] - g_al.u[i][1]);
	}
	*cnt = m;
	if (!m) return 1e9;
	qsort(e, (size_t)m, sizeof e[0], al_cmp);
	return e[m / 2];
}

/* 6 x 6 symmetric positive definite: Cholesky */
static int al_chol(double A[36], const double b[6], double x[6])
{
	for (int j = 0; j < 6; j++) {
		double s = A[6 * j + j];
		for (int k = 0; k < j; k++) s -= A[6 * j + k] * A[6 * j + k];
		if (s <= 0) return -1;
		A[6 * j + j] = sqrt(s);
		for (int i = j + 1; i < 6; i++) {
			double r = A[6 * i + j];
			for (int k = 0; k < j; k++) r -= A[6 * i + k] * A[6 * j + k];
			A[6 * i + j] = r / A[6 * j + j];
		}
	}
	double y[6];
	for (int i = 0; i < 6; i++) {
		double r = b[i];
		for (int k = 0; k < i; k++) r -= A[6 * i + k] * y[k];
		y[i] = r / A[6 * i + i];
	}
	for (int i = 5; i >= 0; i--) {
		double r = y[i];
		for (int k = i + 1; k < 6; k++) r -= A[6 * k + i] * x[k];
		x[i] = r / A[6 * i + i];
	}
	return 0;
}


/* ---- the background arbiter: face points constrain the pose only at the face's depth (a rotation and a
 * translation of Z times it cancel there: measured 2026-11-18, two face fits 20.6 px apart at 0.7 m, 113.7 px at
 * 2.5 m). The static background settles that direction: NCC of blurred gradient magnitude between the webcam
 * frame and the Kinect's colour carried into the webcam through the point map, over points >= 1.2 m away.
 * Gradients, not grey levels: the two sensors' colours differ (what sank the flow lock). */
static uint8_t *g_lastk, *g_lastw; /* newest frames of each camera, RGB24 */
static float *g_bgk, *g_bgw, *g_bgtmp, *g_bgpm;

static void al_blur(float *a, float *tmp)
{
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			const float *r = a + (size_t)y * W;
			const int x0 = x > 1 ? x - 2 : 0, x1 = x > 0 ? x - 1 : 0, x3 = x < W - 1 ? x + 1 : W - 1, x4 = x < W - 2 ? x + 2 : W - 1;
			tmp[(size_t)y * W + x] = (r[x0] + 4 * r[x1] + 6 * r[x] + 4 * r[x3] + r[x4]) * (1.0f / 16);
		}
	for (int y = 0; y < H; y++) {
		const int y0 = y > 1 ? y - 2 : 0, y1 = y > 0 ? y - 1 : 0, y3 = y < H - 1 ? y + 1 : H - 1, y4 = y < H - 2 ? y + 2 : H - 1;
		for (int x = 0; x < W; x++)
			a[(size_t)y * W + x] = (tmp[(size_t)y0 * W + x] + 4 * tmp[(size_t)y1 * W + x] + 6 * tmp[(size_t)y * W + x] + 4 * tmp[(size_t)y3 * W + x] +
						tmp[(size_t)y4 * W + x]) * (1.0f / 16);
	}
}

/* grey, blurred twice, Sobel magnitude, then `post` blurs: edges a few pixels wide, so a pose a little off still overlaps */
static void al_grad(const uint8_t *rgb, float *out, float *g, int post)
{
	for (int i = 0; i < W * H; i++) g[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
	al_blur(g, out);
	al_blur(g, out);
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			if (!x || !y || x == W - 1 || y == H - 1) { out[(size_t)y * W + x] = 0; continue; }
			const float *a = g + (size_t)(y - 1) * W + x, *b = g + (size_t)y * W + x, *c = g + (size_t)(y + 1) * W + x;
			const float gx = (a[1] + 2 * b[1] + c[1]) - (a[-1] + 2 * b[-1] + c[-1]), gy = (c[-1] + 2 * c[0] + c[1]) - (a[-1] + 2 * a[0] + a[1]);
			out[(size_t)y * W + x] = sqrtf(gx * gx + gy * gy);
		}
	for (int i = 0; i < post; i++) al_blur(out, g); /* post blurs: 6 = a wide basin, 2 = the precise peak (alscape, 2026-11-18) */
}

static int al_bg_prepare(int post)
{
	if (!g_lastk || !g_lastw) return -1;
	if (!g_bgk) {
		g_bgk = malloc(sizeof(float) * W * H);
		g_bgw = malloc(sizeof(float) * W * H);
		g_bgtmp = malloc(sizeof(float) * W * H);
		g_bgpm = malloc(sizeof(float) * 4 * W * H);
		if (!g_bgk || !g_bgw || !g_bgtmp || !g_bgpm) return -1;
	}
	al_grad(g_lastk, g_bgk, g_bgtmp, post);
	al_grad(g_lastw, g_bgw, g_bgtmp, post);
	pmap_fresh();
	glGetTextureImage(t_pmap, 0, GL_RGBA, GL_FLOAT, (GLsizei)(sizeof(float) * 4 * W * H), g_bgpm);
	return 0;
}

static double al_bg_ncc(const double R[9], const double t[3], int *cnt)
{
	double n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, p[2];
	for (int y = 2; y < H - 2; y += 2)
		for (int x = 2; x < W - 2; x += 2) {
			const float *P = g_bgpm + 4 * ((size_t)y * W + x);
			if (P[3] < 0.5f || P[2] < 1200.0f) continue;
			const double X[3] = {P[0], P[1], P[2]};
			al_project(R, t, X, p);
			if (p[0] < 1 || p[1] < 1 || p[0] >= W - 2 || p[1] >= H - 2) continue;
			const int u = (int)p[0], v = (int)p[1];
			const double fu = p[0] - u, fv = p[1] - v;
			const float *r0 = g_bgw + (size_t)v * W + u, *r1 = r0 + W;
			const double b = (1 - fv) * ((1 - fu) * r0[0] + fu * r0[1]) + fv * ((1 - fu) * r1[0] + fu * r1[1]), a = g_bgk[(size_t)y * W + x];
			n++; sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
		}
	*cnt = (int)n;
	const double den = (n * saa - sa * sa) * (n * sbb - sb * sb);
	return n > 100 && den > 0 ? (n * sab - sa * sb) / sqrt(den) : -1.0;
}

/* 6 x 6 symmetric: eigenvalues and vectors (columns of V) by cyclic Jacobi */
static void al_eig6(const double A0[36], double ev[6], double V[36])
{
	double A[36];
	memcpy(A, A0, sizeof A);
	for (int i = 0; i < 36; i++) V[i] = i % 7 == 0;
	for (int sweep = 0; sweep < 50; sweep++) {
		double off = 0;
		for (int i = 0; i < 6; i++) for (int j = i + 1; j < 6; j++) off += A[6 * i + j] * A[6 * i + j];
		if (off < 1e-24) break;
		for (int p = 0; p < 6; p++)
			for (int q = p + 1; q < 6; q++) {
				if (fabs(A[6 * p + q]) < 1e-300) continue;
				const double th = 0.5 * (A[6 * q + q] - A[6 * p + p]) / A[6 * p + q];
				const double tt = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0)), c = 1.0 / sqrt(tt * tt + 1.0), s = tt * c;
				for (int k = 0; k < 6; k++) { /* A <- J^T A J */
					const double akp = A[6 * k + p], akq = A[6 * k + q];
					A[6 * k + p] = c * akp - s * akq;
					A[6 * k + q] = s * akp + c * akq;
				}
				for (int k = 0; k < 6; k++) {
					const double apk = A[6 * p + k], aqk = A[6 * q + k];
					A[6 * p + k] = c * apk - s * aqk;
					A[6 * q + k] = s * apk + c * aqk;
				}
				for (int k = 0; k < 6; k++) {
					const double vkp = V[6 * k + p], vkq = V[6 * k + q];
					V[6 * k + p] = c * vkp - s * vkq;
					V[6 * k + q] = s * vkp + c * vkq;
				}
			}
	}
	for (int i = 0; i < 6; i++) ev[i] = A[7 * i];
}

/* the joint objective at a pose: minus the background NCC, plus 0.05 per px the face's train median exceeds the
 * LM face fit's by more than 1.5 px (10 px worse on the face costs 0.43 of NCC) */
static double al_objective(const double R[9], const double t[3], double face_best, double *fm, double *nc)
{
	int c1, c2;
	*fm = al_median(R, t, 0, &c1);
	*nc = al_bg_ncc(R, t, &c2);
	return -*nc + (*fm > face_best + 1.5 ? 0.05 * (*fm - face_best - 1.5) : 0.0);
}

static void al_pose_at(const double Rb[9], const double tb[3], const double x[6], double zf, double R[9], double t[3])
{
	const double w[3] = {x[0] / zf, x[1] / zf, x[2] / zf};
	memcpy(R, Rb, sizeof(double) * 9);
	al_rotate(R, w);
	for (int k = 0; k < 3; k++) t[k] = tb[k] + x[3 + k];
}

/* Nelder-Mead over x = (rotation x zf, translation mm) around (R, t); the result replaces (R, t) */
static void al_nm(double R[9], double t[3], double zf, double face_best, double step)
{
	double Rb[9], tb[3], s[7][6], f[7], Rp[9], tp[3], fm, nc;
	memcpy(Rb, R, sizeof Rb);
	memcpy(tb, t, sizeof tb);
	for (int v = 0; v < 7; v++) {
		for (int k = 0; k < 6; k++) s[v][k] = v == k + 1 ? step : 0.0;
		al_pose_at(Rb, tb, s[v], zf, Rp, tp);
		f[v] = al_objective(Rp, tp, face_best, &fm, &nc);
	}
	for (int it = 0; it < 300; it++) {
		for (int i = 1; i < 7; i++) /* order: f[0] best */
			for (int j = i; j > 0 && f[j] < f[j - 1]; j--) {
				double sv[6], ft = f[j];
				memcpy(sv, s[j], sizeof sv); memcpy(s[j], s[j - 1], sizeof sv); memcpy(s[j - 1], sv, sizeof sv);
				f[j] = f[j - 1]; f[j - 1] = ft;
			}
		double size = 0;
		for (int v = 1; v < 7; v++) for (int k = 0; k < 6; k++) size = fmax(size, fabs(s[v][k] - s[0][k]));
		if (size < 0.25) break;
		double c[6] = {0}, xr[6], xe[6], xc[6], fr, fe, fc;
		for (int v = 0; v < 6; v++) for (int k = 0; k < 6; k++) c[k] += s[v][k] / 6.0;
		for (int k = 0; k < 6; k++) xr[k] = c[k] + (c[k] - s[6][k]);
		al_pose_at(Rb, tb, xr, zf, Rp, tp);
		fr = al_objective(Rp, tp, face_best, &fm, &nc);
		if (fr < f[0]) {
			for (int k = 0; k < 6; k++) xe[k] = c[k] + 2.0 * (c[k] - s[6][k]);
			al_pose_at(Rb, tb, xe, zf, Rp, tp);
			fe = al_objective(Rp, tp, face_best, &fm, &nc);
			if (fe < fr) { memcpy(s[6], xe, sizeof xe); f[6] = fe; } else { memcpy(s[6], xr, sizeof xr); f[6] = fr; }
		} else if (fr < f[5]) {
			memcpy(s[6], xr, sizeof xr); f[6] = fr;
		} else {
			for (int k = 0; k < 6; k++) xc[k] = c[k] + 0.5 * (s[6][k] - c[k]);
			al_pose_at(Rb, tb, xc, zf, Rp, tp);
			fc = al_objective(Rp, tp, face_best, &fm, &nc);
			if (fc < f[6]) { memcpy(s[6], xc, sizeof xc); f[6] = fc; }
			else
				for (int v = 1; v < 7; v++) { /* shrink toward the best */
					for (int k = 0; k < 6; k++) s[v][k] = s[0][k] + 0.5 * (s[v][k] - s[0][k]);
					al_pose_at(Rb, tb, s[v], zf, Rp, tp);
					f[v] = al_objective(Rp, tp, face_best, &fm, &nc);
				}
		}
	}
	int b = 0;
	for (int v = 1; v < 7; v++) if (f[v] < f[b]) b = v;
	al_pose_at(Rb, tb, s[b], zf, R, t);
}
static void al_solve(void)
{
	double R0[9], t0[3], R[9], t[3];
	for (int i = 0; i < 9; i++) R0[i] = g_cal.R[i];
	for (int i = 0; i < 3; i++) t0[i] = g_cal.t[i];
	memcpy(R, R0, sizeof R);
	memcpy(t, t0, sizeof t);
	double lambda = 1e-3, cost = al_cost(R, t, 0), A_last[36] = {0};
	for (int it = 0; it < 30; it++) {
		double A[36] = {0}, g[6] = {0}, p[2], q[2], J[2][6];
		for (int i = 0; i < g_al.n; i++) {
			if (al_block(i) != 0) continue;
			al_project(R, t, g_al.X[i], p);
			const double r0 = p[0] - g_al.u[i][0], r1 = p[1] - g_al.u[i][1], e = hypot(r0, r1), wt = e <= 3.0 ? 1.0 : 3.0 / e;
			for (int k = 0; k < 6; k++) { /* central differences: 1e-6 rad, 1e-3 mm */
				double Rp[9], tp[3], Rm[9], tm[3], d[3] = {0, 0, 0};
				const double h = k < 3 ? 1e-6 : 1e-3;
				memcpy(Rp, R, sizeof Rp); memcpy(tp, t, sizeof tp); memcpy(Rm, R, sizeof Rm); memcpy(tm, t, sizeof tm);
				if (k < 3) { d[k] = h; al_rotate(Rp, d); d[k] = -h; al_rotate(Rm, d); }
				else { tp[k - 3] += h; tm[k - 3] -= h; }
				al_project(Rp, tp, g_al.X[i], p);
				al_project(Rm, tm, g_al.X[i], q);
				J[0][k] = (p[0] - q[0]) / (2 * h);
				J[1][k] = (p[1] - q[1]) / (2 * h);
			}
			for (int a = 0; a < 6; a++) {
				g[a] -= wt * (J[0][a] * r0 + J[1][a] * r1);
				for (int b = 0; b < 6; b++) A[6 * a + b] += wt * (J[0][a] * J[0][b] + J[1][a] * J[1][b]);
			}
		}
		memcpy(A_last, A, sizeof A_last);
		double Ad[36], x[6];
		memcpy(Ad, A, sizeof Ad);
		for (int a = 0; a < 6; a++) Ad[7 * a] += lambda * (A[7 * a] > 0 ? A[7 * a] : 1.0);
		if (al_chol(Ad, g, x)) { lambda *= 10; continue; }
		double Rn[9], tn[3];
		memcpy(Rn, R, sizeof Rn);
		al_rotate(Rn, x);
		for (int k = 0; k < 3; k++) tn[k] = t[k] + x[3 + k];
		const double cn = al_cost(Rn, tn, 0);
		if (cn < cost) {
			memcpy(R, Rn, sizeof R);
			memcpy(t, tn, sizeof t);
			const double step = fabs(x[0]) + fabs(x[1]) + fabs(x[2]) + 1e-3 * (fabs(x[3]) + fabs(x[4]) + fabs(x[5]));
			cost = cn;
			lambda = fmax(lambda / 3, 1e-9);
			if (step < 1e-9) break;
		} else lambda *= 4;
	}
	int ntr, nte, ntr0;
	g_al.solves++;
	(void)al_median(R0, t0, 0, &ntr0);
	g_al.old_med = al_median(R0, t0, 1, &g_al.n_test);
	/* joint refinement. The face pins the near field, the static background the far one, which breaks the valley
	 * the face alone leaves (dt ~ Z_face x dtheta). Nelder-Mead over the 6 DOF maximising the background's
	 * gradient NCC with the face held (al_objective); coarse (6 post blurs: wide basin) then fine (2: the precise
	 * peak), from the LM face fit and from the pose in use; the better end wins. */
	double zf = 0;
	for (int i = 0; i < g_al.n; i++) zf += g_al.X[i][2];
	zf = g_al.n ? zf / g_al.n : 700.0;
	{ /* how ill-posed the face alone is: condition of J^T J, rotation scaled by the face depth */
		double As[36], ev[6], V[36], lo = 1e300, hi = 0;
		for (int a = 0; a < 6; a++)
			for (int b = 0; b < 6; b++) As[6 * a + b] = A_last[6 * a + b] / ((a < 3 ? zf : 1.0) * (b < 3 ? zf : 1.0));
		al_eig6(As, ev, V);
		for (int i = 0; i < 6; i++) { lo = fmin(lo, ev[i]); hi = fmax(hi, ev[i]); }
		printf("align: face-only problem: condition %.0f (eigenvalues %.3g .. %.3g, rotation scaled by %.0f mm)\n", hi / fmax(lo, 1e-30), lo, hi, zf);
	}
	const double med_lm = al_median(R, t, 1, &nte), face_best = al_median(R, t, 0, &ntr);
	int nbg = 0, have_bg = !al_bg_prepare(2);
	const double ncc_old = have_bg ? al_bg_ncc(R0, t0, &nbg) : -2, ncc_lm = have_bg ? al_bg_ncc(R, t, &nbg) : -2;
	if (have_bg) {
		double bR[9], bt[3], bf = 1e9, fm, nc;
		for (int start = 0; start < 2; start++) {
			double Rs[9], ts[3];
			memcpy(Rs, start ? R0 : R, sizeof Rs);
			memcpy(ts, start ? t0 : t, sizeof ts);
			for (int stage = 0; stage < 2; stage++) {
				al_bg_prepare(stage ? 2 : 6);
				al_nm(Rs, ts, zf, face_best, stage ? 10.0 : 40.0);
			}
			const double f = al_objective(Rs, ts, face_best, &fm, &nc);
			printf("align: joint refinement from the %s: background NCC %.3f, face train median %.2f px\n", start ? "pose in use" : "LM face fit", nc, fm);
			if (f < bf) { bf = f; memcpy(bR, Rs, sizeof bR); memcpy(bt, ts, sizeof bt); }
		}
		memcpy(R, bR, sizeof bR);
		memcpy(t, bt, sizeof bt);
	}
	g_al.train_rms = al_median(R, t, 0, &ntr);
	g_al.new_med = al_median(R, t, 1, &nte);
	const double ncc_new = have_bg ? al_bg_ncc(R, t, &nbg) : -2; /* the last stage left the 2-blur gradients in place */
	double tr = 0; /* rotation between the poses: angle of R R0^T */
	for (int i = 0; i < 3; i++)
		for (int k = 0; k < 3; k++) tr += R[3 * i + k] * R0[3 * i + k];
	g_al.d_rot = acos(fmax(-1.0, fmin(1.0, 0.5 * (tr - 1.0)))) * 57.29577951308232;
	g_al.d_t = sqrt((t[0] - t0[0]) * (t[0] - t0[0]) + (t[1] - t0[1]) * (t[1] - t0[1]) + (t[2] - t0[2]) * (t[2] - t0[2]));
	/* better on the face by 10 %, or as good on it (1 px) and better on the background (NCC + 0.02); never worse on the background */
	const int ok = nte >= 100 && g_al.spread >= 150.0 && have_bg && ncc_new >= ncc_old - 0.02 &&
		       (g_al.new_med < 0.9 * g_al.old_med || (g_al.new_med <= g_al.old_med + 1.0 && ncc_new > ncc_old + 0.02));
	printf("align: solve %u: %d train %d test corr, head travel %.0f mm; test median %.2f px in use -> %.2f px LM-only -> %.2f px new (train %.2f); "
	       "background NCC (%d pts >= 1.2 m) %.3f in use, %.3f LM-only, %.3f new; change %.2f deg %.1f mm: %s\n", g_al.solves, ntr, nte, g_al.spread,
	       g_al.old_med, med_lm, g_al.new_med, g_al.train_rms, nbg, ncc_old, ncc_lm, ncc_new, g_al.d_rot, g_al.d_t, ok ? "APPLIED" : "kept the pose in use");
	g_al.ncc_old = ncc_old;
	g_al.ncc_new = ncc_new;
	if (!ok) return;
	for (int i = 0; i < 9; i++) g_cal.R[i] = (float)R[i];
	for (int i = 0; i < 3; i++) g_cal.t[i] = (float)t[i];
	apply_calib();
	g_al.applied++;
}

/* every loop pass, after the face results. All three quantities of a correspondence refer to one instant, the
 * newest depth frame's (t_ref): the Kinect's landmarks are carried there by its tracker's rate before the point
 * map is read, the webcam's by its own. Gates: rate < 120 px/s (so a carry of <= 100 ms moves <= 12 px), and each
 * detection within 4 px (every landmark) of where its camera's previous one, carried by the rate, said it would
 * be: a rate that does not describe the face is caught there. */
static int al_consistent(const vtf_out *o, int c)
{
	const double dt = o->t_face - g_al.t_prev[c];
	const int ok = g_al.t_prev[c] > 0 && dt > 0 && dt < 0.25;
	for (int l = 0; ok && l < 5; l++)
		if (fabs(o->lm[2 * l] - (g_al.lm_prev[c][2 * l] + o->rate_u * dt)) > 4.0 || fabs(o->lm[2 * l + 1] - (g_al.lm_prev[c][2 * l + 1] + o->rate_v * dt)) > 4.0)
			return 0;
	return ok;
}

static void align_step(void)
{
	if (!g_align || !g_cal.ok || !g_vtf[0] || !g_vtf[1]) return;
	if (g_room.moving) { /* the tilt motor turns the Kinect against the webcam: what was gathered no longer holds */
		if (g_al.n) { g_al.n = g_al.last_solve_n = 0; g_al.kept = 0; g_al.resets++; }
		return;
	}
	for (int c = 0; c < 2; c++) {
		const vtf_out *o = &g_fo[c];
		if (o->face_seq == g_al.seq[c]) continue;
		g_al.seq[c] = o->face_seq;
		const int ok = hypotf(o->rate_u, o->rate_v) < 120.0f && al_consistent(o, c);
		memcpy(g_al.lm_prev[c], o->lm, sizeof o->lm);
		g_al.t_prev[c] = o->t_face;
		if (!ok) { g_al.unstill++; continue; }
		if (c == 0) {
			pmap_fresh();
			const double dt = g_t_depth - o->t_face; /* the point map is of the newest depth frame */
			if (fabs(dt) > 0.1) { g_al.untimed++; continue; }
			face5 f;
			memset(&f, 0, sizeof f);
			memcpy(f.box, o->face, sizeof f.box);
			for (int l = 0; l < 5; l++) { f.lm[2 * l] = o->lm[2 * l] + o->rate_u * (float)dt; f.lm[2 * l + 1] = o->lm[2 * l + 1] + o->rate_v * (float)dt; }
			float X[15];
			if (face_3d(&f, X) < 3) continue;
			for (int l = 0; l < 5; l++) {
				g_al.okk[l] = isfinite(X[3 * l]) && X[3 * l + 2] > 300.0f;
				memcpy(g_al.Xk[l], X + 3 * l, sizeof g_al.Xk[l]);
			}
			g_al.tk = g_t_depth;
			g_al.have_k = 1;
			g_al.still_k++;
		} else {
			memcpy(g_al.lmw, o->lm, sizeof g_al.lmw);
			g_al.rw[0] = o->rate_u;
			g_al.rw[1] = o->rate_v;
			g_al.tw = o->t_face;
			g_al.have_w = 1;
			g_al.still_w++;
		}
	}
	if (!g_al.have_k || !g_al.have_w) return;
	const double dw = g_al.tk - g_al.tw; /* carry the webcam's landmarks to the depth frame's instant */
	if (fabs(dw) > 0.1) {
		g_al.untimed++;
		if (g_al.tk < g_al.tw) g_al.have_k = 0; else g_al.have_w = 0;
		return;
	}
	for (int l = 0; l < 5; l++) { g_al.uw[l][0] = g_al.lmw[2 * l] + g_al.rw[0] * (float)dw; g_al.uw[l][1] = g_al.lmw[2 * l + 1] + g_al.rw[1] * (float)dw; }
	g_al.have_k = g_al.have_w = 0;
	g_al.pairs++;
	if (!g_al.okk[2]) return;
	const double nose[3] = {g_al.Xk[2][0], g_al.Xk[2][1], g_al.Xk[2][2]}, t = g_al.tk;
	const double mv = sqrt((nose[0] - g_al.nose_acc[0]) * (nose[0] - g_al.nose_acc[0]) + (nose[1] - g_al.nose_acc[1]) * (nose[1] - g_al.nose_acc[1]) +
			       (nose[2] - g_al.nose_acc[2]) * (nose[2] - g_al.nose_acc[2]));
	if (g_al.kept && mv < 15.0 && t - g_al.t_acc < 1.0) return;
	g_al.kept++;
	g_al.t_acc = t;
	memcpy(g_al.nose_acc, nose, sizeof g_al.nose_acc);
	g_al.spread = 0;
	for (int k = 0; k < 3; k++) {
		if (g_al.kept == 1 || nose[k] < g_al.lo[k]) g_al.lo[k] = nose[k];
		if (g_al.kept == 1 || nose[k] > g_al.hi[k]) g_al.hi[k] = nose[k];
		g_al.spread = fmax(g_al.spread, g_al.hi[k] - g_al.lo[k]);
	}
	for (int l = 0; l < 5 && g_al.n < AL_MAX; l++) {
		if (!g_al.okk[l]) continue;
		for (int k = 0; k < 3; k++) g_al.X[g_al.n][k] = g_al.Xk[l][k];
		g_al.u[g_al.n][0] = g_al.uw[l][0];
		g_al.u[g_al.n][1] = g_al.uw[l][1];
		g_al.t[g_al.n] = t;
		g_al.n++;
	}
	if (g_al.n - g_al.last_solve_n >= 150 && g_al.spread >= 150.0) { /* less head travel than that: the pose is not determined */
		g_al.last_solve_n = g_al.n;
		al_solve();
	}
}

static void cam_reset(void)
{
	cam.yaw = cam.pitch = 0.0f;
	cam.dist = 1500.0f; /* pivot 1.5 m in front; eye at the Kinect: the view equals the depth image */
	cam.px = cam.py = 0.0f;
	cam.pz = -1500.0f;
}

static mat4 cam_rot(void) { return mmul(mrotx(cam.pitch), mroty(cam.yaw)); }

static mat4 cam_mvp(int fbw, int fbh)
{
	float fovy = 2.0f * atanf(240.0f * g_k); /* the depth camera's own vertical field of view */
	mat4 v = mmul(mtrans(0.0f, 0.0f, -cam.dist), mmul(cam_rot(), mtrans(-cam.px, -cam.py, -cam.pz)));
	return mmul(mpersp(fovy, (float)fbw / (float)(fbh > 0 ? fbh : 1), 50.0f, 30000.0f), v);
}

static void draw_mesh(int m, const mat4 *mvp)
{
	glProgramUniformMatrix4fv(p_mesh[m], loc_mvp[m], 1, GL_FALSE, mvp->m);
	glProgramUniform1f(p_mesh[m], loc_k[m], g_k);
	glProgramUniform1f(p_mesh[m], loc_cull[m], g_cull);
	glProgramUniform1i(p_mesh[m], loc_occl[m], g_occl);
	glProgramUniform1f(p_mesh[m], loc_tshift[m], g_tshift);
	glProgramUniformMatrix3fv(p_mesh[m], loc_rcam[m], 1, GL_TRUE, g_room.Rnow);
	glUseProgram(p_mesh[m]);
	glBindVertexArray(vao[m]);
	if (m) glDrawArrays(GL_TRIANGLES, 0, NVERT);
	else glDrawElements(GL_TRIANGLES, NVERT, GL_UNSIGNED_INT, NULL);
}

/* two 320x240 panels at the bottom left: Kinect, then webcam */
static void draw_pip(int fbw, int fbh)
{
	GLuint src[2] = {
		g_pip == 2 ? (g_fsk ? t_fsk : g_fk && flow_ready(g_fk) ? flow_result(g_fk) : 0) : g_show_k,
		g_pip == 2 ? (g_fsw ? t_fsw : g_fw && flow_ready(g_fw) ? flow_result(g_fw) : 0) : (g_cam ? g_show_w : 0),
	};
	glDisable(GL_DEPTH_TEST);
	glUseProgram(p_pip);
	glBindVertexArray(vao[1]);
	glProgramUniform1i(p_pip, loc_pip_mode, g_pip == 2);
	for (int i = 0; i < 2; i++) {
		if (!src[i]) continue;
		const vtf_out *fo = &g_fo[i];
		const int boxed = g_vtf[i] && (fo->state == VTF_NEW || fo->state == VTF_OBSERVED || fo->state == VTF_UNOBSERVED);
		glProgramUniform4f(p_pip, loc_pip_box, boxed ? fo->cx - 0.5f * fo->w : 0.0f, boxed ? fo->cy - 0.5f * fo->h : 0.0f,
				   boxed ? fo->cx + 0.5f * fo->w : 0.0f, boxed ? fo->cy + 0.5f * fo->h : 0.0f);
		glProgramUniform3f(p_pip, loc_pip_boxcol, fo->state == VTF_UNOBSERVED ? 1.0f : 0.2f, fo->state == VTF_NEW ? 0.8f : fo->state == VTF_UNOBSERVED ? 0.85f : 1.0f,
				   fo->state == VTF_NEW ? 1.0f : 0.2f);
		glProgramUniform2fv(p_pip, loc_pip_feat, 64, &fo->feat[0][0]);
		glProgramUniform1i(p_pip, loc_pip_nfeat, boxed ? fo->nshow : 0);
		glProgramUniform1i(p_pip, loc_pip_nmem, (int)fo->members);
		float x0 = 8.0f + 328.0f * (float)i, y0 = 8.0f, x1 = x0 + 320.0f, y1 = y0 + 240.0f;
		glProgramUniform4f(p_pip, loc_pip_rect, 2.0f * x0 / (float)fbw - 1.0f, 2.0f * y0 / (float)fbh - 1.0f,
				   2.0f * x1 / (float)fbw - 1.0f, 2.0f * y1 / (float)fbh - 1.0f);
		glBindTextureUnit(5, src[i]);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
	glEnable(GL_DEPTH_TEST);
}

/* --dump-flow: gray pair (8-bit PGM, the exact values the GPU flow saw) + flow result (RGBA32F raw) */
static int dump_flow(const char *dir, flowcam *f, const char *name)
{
	float *g = malloc(sizeof(float) * NPIX * 4);
	uint8_t *b = malloc(NPIX);
	char path[1024];
	int err = 0;
	if (!g || !b) return 1;
	glFinish();
	for (int k = 0; k < 2; k++) {
		glGetTextureImage(flow_gray(f, k), 0, GL_RED, GL_FLOAT, sizeof(float) * NPIX, g);
		for (int i = 0; i < NPIX; i++) b[i] = (uint8_t)lrintf(fminf(fmaxf(g[i], 0.0f), 255.0f));
		snprintf(path, sizeof path, "%s/%s-%s.pgm", dir, name, k ? "newest" : "previous");
		FILE *o = fopen(path, "wb");
		err |= !o || fprintf(o, "P5\n%d %d\n255\n", W, H) < 0 || fwrite(b, 1, NPIX, o) != NPIX || fclose(o);
	}
	glGetTextureImage(flow_result(f), 0, GL_RGBA, GL_FLOAT, sizeof(float) * NPIX * 4, g);
	snprintf(path, sizeof path, "%s/%s-flow.f32", dir, name);
	FILE *o = fopen(path, "wb");
	err |= !o || fwrite(g, sizeof(float) * 4, NPIX, o) != NPIX || fclose(o);
	unsigned valid = 0;
	for (int i = 0; i < NPIX; i++) valid += g[4 * i + 3] > 0.5f;
	printf("dump %s: dt %.4f s, valid %u of %d (%.1f %%), GPU %.3f ms per frame%s\n", name, flow_dt(f), valid, NPIX,
	       100.0 * valid / NPIX, flow_gpu_ms(f), err ? ", WRITE ERROR" : "");
	free(g);
	free(b);
	return err;
}

/* ==== input ===================================================================================== */
static void on_key(GLFWwindow *w, int key, int sc, int action, int mods)
{
	(void)sc; (void)mods;
	if (action == GLFW_RELEASE) return;
	switch (key) {
	case GLFW_KEY_ESCAPE: case GLFW_KEY_Q: glfwSetWindowShouldClose(w, 1); break;
	case GLFW_KEY_R: cam_reset(); break;
	case GLFW_KEY_M: g_method ^= 1; break;
	case GLFW_KEY_O: g_occl = (g_occl + 1) % (g_cal.ok ? 4 : 2); break;
	case GLFW_KEY_LEFT_BRACKET: g_cull = fmaxf(g_cull * 0.8f, 0.002f); break;
	case GLFW_KEY_RIGHT_BRACKET: g_cull = fminf(g_cull * 1.25f, 2.0f); break;
	case GLFW_KEY_W: tilt_deg = tilt_deg + 2 > 27 ? 27 : tilt_deg + 2; freenect_set_tilt_degs(fdev, tilt_deg); break;
	case GLFW_KEY_X: tilt_deg = tilt_deg - 2 < -27 ? -27 : tilt_deg - 2; freenect_set_tilt_degs(fdev, tilt_deg); break;
	case GLFW_KEY_S: tilt_deg = 0; freenect_set_tilt_degs(fdev, 0); break;
	case GLFW_KEY_N: g_pip = (g_pip + 1) % 3; break;
	case GLFW_KEY_V: g_arrows ^= 1; break;
	case GLFW_KEY_G: if (p_fg) g_fg ^= 1; break;
	case GLFW_KEY_T: g_geo ^= 1; break;
	case GLFW_KEY_F: g_dfilter ^= 1; break;
	case GLFW_KEY_L: if (g_flock) g_lock ^= 1; break;
	case GLFW_KEY_K:
		if (g_cal.ok && g_cal.path[0]) printf("kinect3d: %s %s\n", save_calib(g_cal.path) ? "could not save" : "saved the locked pose to", g_cal.path);
		break;
	default: break;
	}
}

static void on_button(GLFWwindow *w, int button, int action, int mods)
{
	(void)mods;
	if (action == GLFW_PRESS) {
		drag = button == GLFW_MOUSE_BUTTON_LEFT ? 1 : 2;
		glfwGetCursorPos(w, &mx_last, &my_last);
	} else {
		drag = 0;
	}
}

static void on_cursor(GLFWwindow *w, double x, double y)
{
	float dx = (float)(x - mx_last), dy = (float)(y - my_last);
	(void)w;
	mx_last = x;
	my_last = y;
	if (drag == 1) {
		cam.yaw += dx * 0.005f;
		cam.pitch = fmaxf(-1.55f, fminf(1.55f, cam.pitch + dy * 0.005f));
	} else if (drag == 2) {
		mat4 r = cam_rot(); /* rows of r = camera right, up, back in world coordinates */
		float s = cam.dist * 0.0012f;
		cam.px += (-r.m[0] * dx + r.m[1] * dy) * s;
		cam.py += (-r.m[4] * dx + r.m[5] * dy) * s;
		cam.pz += (-r.m[8] * dx + r.m[9] * dy) * s;
	}
}

static void on_scroll(GLFWwindow *w, double xo, double yo)
{
	(void)w; (void)xo;
	cam.dist = fmaxf(100.0f, fminf(20000.0f, cam.dist * powf(0.9f, (float)yo)));
}

/* ==== helpers =================================================================================== */
static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmp_u64(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return (x > y) - (x < y);
}

/* wait for both streams; restart the video stream if RGB stays silent (first open after gspca_kinect) */
static int wait_first_frames(GLFWwindow *win, const uint8_t **d, const uint8_t **v)
{
	double t0 = now_s(), tv = t0;
	int restarts = 0;
	*d = *v = NULL;
	while (!*d || !*v) {
		glfwPollEvents();
		if (glfwWindowShouldClose(win)) return -1;
		if (!*d) *d = tb_acquire(&tb_depth);
		if (!*v) *v = tb_acquire(&tb_rgb);
		double t = now_s();
		if (!*v && t - tv > 3.0 && restarts < 3) {
			fprintf(stderr, "kinect3d: no RGB frame in 3 s, restarting the video stream (%d/3)\n", ++restarts);
			atomic_store(&k_restart_video, 1);
			tv = t;
		}
		if (t - t0 > 15.0) {
			fprintf(stderr, "kinect3d: no %s frames after 15 s\n", !*d ? "depth" : "RGB");
			return -1;
		}
		usleep(5000);
	}
	return 0;
}

/* ==== selftest ================================================================================== */
static int selftest(GLFWwindow *win, const uint16_t *depth, const uint8_t *rgb, const freenect_registration *reg)
{
	int fails = 0;
	uint8_t *ref = malloc(NPIX * 3), *gpu = malloc(NPIX * 4);
	if (!ref || !gpu) return 1;

	/* 1. GPU registration + occlusion vs libfreenect's freenect_map_rgb_to_depth on the same frame */
	for (int i = 0; i < NPIX; i++) { ref[3 * i] = 255; ref[3 * i + 1] = 0; ref[3 * i + 2] = 255; }
	freenect_map_rgb_to_depth(fdev, (uint16_t *)depth, (uint8_t *)rgb, ref);
	glBindImageTexture(1, t_out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
	glUseProgram(p_check);
	glDispatchCompute(W / 16, H / 16, 1);
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
	glGetTextureImage(t_out, 0, GL_RGBA, GL_UNSIGNED_BYTE, NPIX * 4, gpu);
	unsigned undefined = 0, mism = 0, black = 0, occl = 0, colour = 0;
	int32_t off = (int32_t)reg->reg_pad_info.start_lines * H;
	for (int i = 0; i < NPIX; i++) {
		unsigned wz = depth[i];
		if (wz >= ZMAX) { undefined++; continue; } /* C reads past depth_to_rgb_shift[] */
		if (wz) { /* C indexes outside its z-buffer only when cx is in bounds and cy is not */
			int32_t s = reg->registration_table[i][0] + reg->depth_to_rgb_shift[wz];
			int32_t cx = s / 256, cy = reg->registration_table[i][1] - off;
			if (cx >= 0 && cx < W && (cy < 0 || cy >= H)) { undefined++; continue; }
		}
		const uint8_t *r = ref + 3 * i, *g = gpu + 4 * i;
		mism += r[0] != g[0] || r[1] != g[1] || r[2] != g[2];
		if (!g[0] && !g[1] && !g[2]) black++;
		else if (g[0] == 255 && !g[1] && g[2] == 255) occl++;
		else colour++;
	}
	printf("%s registration  GPU vs freenect_map_rgb_to_depth: %u of %d pixels differ "
	       "(coloured %u, occluded %u, no colour %u, undefined in C %u)\n",
	       mism ? "FAIL" : "PASS", mism, NPIX - (int)undefined, colour, occl, black, undefined);
	fails += mism != 0 || colour == 0;

	/* 2. back-projection: shader formula in float vs freenect_camera_to_world in double */
	double maxd = 0;
	for (int i = 0; i < NPIX; i++) {
		if (!depth[i] || depth[i] >= ZMAX) continue;
		int x = i % W, y = i / W;
		double wx, wy;
		freenect_camera_to_world(fdev, x, y, depth[i], &wx, &wy);
		float fx = (float)(x - 320) * g_k * (float)depth[i], fy = (float)(y - 240) * g_k * (float)depth[i];
		maxd = fmax(maxd, fmax(fabs(fx - wx), fabs(fy - wy)));
	}
	printf("%s geometry      shader formula vs freenect_camera_to_world: max |diff| %.6f mm\n", maxd < 0.05 ? "PASS" : "FAIL", maxd);
	fails += maxd >= 0.05;

	/* 3. path A vs path B rendered offscreen from an oblique view: must be pixel-identical */
	enum { FW = 1280, FH = 960 };
	GLuint fb, rb[2];
	glCreateFramebuffers(1, &fb);
	glCreateRenderbuffers(2, rb);
	glNamedRenderbufferStorage(rb[0], GL_RGBA8, FW, FH);
	glNamedRenderbufferStorage(rb[1], GL_DEPTH_COMPONENT24, FW, FH);
	glNamedFramebufferRenderbuffer(fb, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[0]);
	glNamedFramebufferRenderbuffer(fb, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb[1]);
	if (glCheckNamedFramebufferStatus(fb, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { printf("FAIL framebuffer incomplete\n"); return fails + 1; }
	uint8_t *img[2] = {malloc((size_t)FW * FH * 4), malloc((size_t)FW * FH * 4)};
	if (!img[0] || !img[1]) return fails + 1;
	cam_reset();
	cam.yaw = 0.5f;
	cam.pitch = 0.3f;
	mat4 mvp = cam_mvp(FW, FH);
	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glViewport(0, 0, FW, FH);
	glEnable(GL_DEPTH_TEST);
	for (int m = 0; m < 2; m++) {
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		draw_mesh(m, &mvp);
		glReadPixels(0, 0, FW, FH, GL_RGBA, GL_UNSIGNED_BYTE, img[m]);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	unsigned diff = 0, covered = 0;
	for (size_t i = 0; i < (size_t)FW * FH; i++) {
		diff += memcmp(img[0] + 4 * i, img[1] + 4 * i, 4) != 0;
		covered += img[1][4 * i + 3] != 0;
	}
	printf("%s mesh paths    A (index + geometry shader) vs B (no buffers): %u of %d pixels differ, mesh covers %.1f %%\n",
	       diff || covered < (unsigned)(FW * FH / 20) ? "FAIL" : "PASS", diff, FW * FH, 100.0 * covered / (FW * FH));
	fails += diff != 0 || covered < (unsigned)(FW * FH / 20);

	glFinish();
	printf("%s GL errors     %u\n", gl_errors ? "FAIL" : "PASS", gl_errors);
	fails += gl_errors != 0;
	printf("%s buffer swaps  triple-buffer protocol mismatches %u\n", atomic_load(&k_bufmiss) ? "FAIL" : "PASS", atomic_load(&k_bufmiss));
	fails += atomic_load(&k_bufmiss) != 0;
	(void)win;
	free(ref); free(gpu); free(img[0]); free(img[1]);
	return fails;
}

/* ==== main ====================================================================================== */
int main(int argc, char **argv)
{
	int do_selftest = 0, bench = 0;
	const char *webcam_dev = NULL, *dump_dir = NULL, *calib_dir = NULL, *calib_file = NULL, *track_dir = NULL, *lock_dump = NULL;
	double track_secs = 25.0, face_secs = 30.0;
	const char *face_dir = NULL, *face_model = NULL;
	int occl_start = -1, do_analyze = 0, stats_n = 0;
	double fg_test = 0; /* --fg-test seconds */
	double face_stats = 0, g_t0_face = 0; /* --face-stats seconds, from the first face */
	const char *align_dump = NULL; /* --dump-align DIR */
	float view_yaw = 0.0f, view_pitch = 0.0f;
	char face_model_default[512];
	snprintf(face_model_default, sizeof face_model_default, "%s/kinect360/models/face_detection_yunet_2023mar.onnx", getenv("HOME") ? getenv("HOME") : ".");
	char calib_default[512];
	snprintf(calib_default, sizeof calib_default, "%s/kinect360/calib/webcam-aligned.txt", getenv("HOME") ? getenv("HOME") : ".");
	{ /* the face-aligned pose when one was saved (the aligner judged it on held-out data), else the calibration */
		FILE *fa = fopen(calib_default, "r");
		if (fa) fclose(fa);
		else snprintf(calib_default, sizeof calib_default, "%s/kinect360/calib/webcam.txt", getenv("HOME") ? getenv("HOME") : ".");
	}
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--dump-shaders") && i + 1 < argc) return dump_shaders(argv[++i]);
		else if (!strcmp(argv[i], "--selftest")) do_selftest = 1;
		else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--method") && i + 1 < argc) g_method = argv[++i][0] == 'b';
		else if (!strcmp(argv[i], "--webcam") && i + 1 < argc) webcam_dev = argv[++i];
		else if (!strcmp(argv[i], "--dump-flow") && i + 1 < argc) dump_dir = argv[++i];
		else if (!strcmp(argv[i], "--calib-dump") && i + 1 < argc) calib_dir = argv[++i];
		else if (!strcmp(argv[i], "--calib") && i + 1 < argc) calib_file = argv[++i];
		else if (!strcmp(argv[i], "--occl") && i + 1 < argc) occl_start = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--analyze")) do_analyze = 1;
		else if (!strcmp(argv[i], "--lock")) g_lock = 1;
		else if (!strcmp(argv[i], "--flow-source") && i + 1 < argc) g_fsrc = strcmp(argv[++i], "lk") != 0;
		else if (!strcmp(argv[i], "--fsr-vec") && i + 1 < argc) { const char *m = argv[++i]; fsr_vector_mode(!strcmp(m, "ema") ? 0 : !strcmp(m, "lk-all") ? 2 : 1); }
		else if (!strcmp(argv[i], "--no-fg")) g_fg = 0;
		else if (!strcmp(argv[i], "--no-geo-time")) g_geo = 0;
		else if (!strcmp(argv[i], "--no-face-track")) g_facetrack = 0;
		else if (!strcmp(argv[i], "--no-align")) g_align = 0;
		else if (!strcmp(argv[i], "--dump-align") && i + 1 < argc) align_dump = argv[++i];
		else if (!strcmp(argv[i], "--face-stats") && i + 1 < argc) face_stats = atof(argv[++i]);
		else if (!strcmp(argv[i], "--fg-test") && i + 1 < argc) fg_test = atof(argv[++i]);
		else if (!strcmp(argv[i], "--scene-stats") && i + 1 < argc) stats_n = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--lock-measure")) g_lock_measure = 1;
		else if (!strcmp(argv[i], "--dump-lock") && i + 1 < argc) { lock_dump = argv[++i]; g_lock_measure = 1; }
		else if (!strcmp(argv[i], "--cull") && i + 1 < argc) g_cull = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--view") && i + 2 < argc) { view_yaw = (float)atof(argv[++i]); view_pitch = (float)atof(argv[++i]); }
		else if (!strcmp(argv[i], "--face-model") && i + 1 < argc) face_model = argv[++i];
		else if (!strcmp(argv[i], "--calib-face") && i + 1 < argc) {
			face_dir = argv[++i];
			if (i + 1 < argc && argv[i + 1][0] != '-') face_secs = atof(argv[++i]);
		}
		else if (!strcmp(argv[i], "--calib-track") && i + 1 < argc) {
			track_dir = argv[++i];
			if (i + 1 < argc && argv[i + 1][0] != '-') track_secs = atof(argv[++i]);
		}
		else {
			fprintf(stderr, "usage: %s [--method a|b] [--selftest] [--bench FRAMES] [--webcam DEV|none]\n"
					"       [--calib FILE|none] [--calib-dump DIR] [--dump-flow DIR] [--dump-shaders DIR]\n", argv[0]);
			return 64;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (kinect_open()) return 1;
	if (strcmp(webcam_dev ? webcam_dev : "", "none")) {
		g_cam = webcam_open(webcam_dev, W, H, 30);
		if (g_cam) printf("kinect3d: webcam %s MJPEG %dx%d at 30 fps\n", webcam_path(g_cam), W, H);
		else printf("kinect3d: running without the webcam\n");
	}

	if (!glfwInit()) { fprintf(stderr, "kinect3d: glfwInit failed\n"); kinect_close(); return 1; }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, do_selftest || bench ? GLFW_TRUE : GLFW_FALSE);
	GLFWwindow *win = glfwCreateWindow(1280, 960, "kinect3d", NULL, NULL);
	if (!win) { fprintf(stderr, "kinect3d: no GL 4.6 core window\n"); glfwTerminate(); kinect_close(); return 1; }
	glfwMakeContextCurrent(win);
	glfwSwapInterval(1);
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(gl_debug, NULL);
	printf("kinect3d: %s, %s\n", (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));
	gl_setup();
	g_fk = flow_create(W, H, "kinect");
	sflow_setup();
	if (g_fsrc || g_fg) { /* 4 pinned cores per camera: 8 threads on 8 cores, the FSR port's own rule */
		static const int ck[4] = {0, 1, 2, 3}, cw[4] = {4, 5, 6, 7};
		g_fsk = fsr_open(W, H, ck, 4);
		if (g_cam) g_fsw = fsr_open(W, H, cw, 4);
		t_fsk = tex2d(GL_RGBA32F, 80, 60, 1);
		t_fsw = tex2d(GL_RGBA32F, 80, 60, 1);
		if (!g_fsk) { fprintf(stderr, "kinect3d: FSR flow engine did not start: GPU LK flow, no frame generation\n"); g_fsrc = 0; g_fg = 0; }
		else printf("kinect3d: FSR 3 optical flow on the Kinect%s, 4 threads each: velocity field from %s, frame generation %s\n",
			    g_fsw ? " and the webcam" : "", g_fsrc ? "FSR" : "LK", g_fg ? "on (G)" : "off (G)");
	}
	if (g_facetrack) { /* the face tracker: VisionTrack per camera, YuNet every 3rd frame (every frame without a region) */
		g_vtf[0] = vtf_open("kinect", W, H, face_model ? face_model : face_model_default, 3);
		if (g_cam) g_vtf[1] = vtf_open("webcam", W, H, face_model ? face_model : face_model_default, 3);
		printf("kinect3d: face tracker (VisionTrack track.c, YuNet keeps it on the face): Kinect %s, webcam %s\n",
		       g_vtf[0] ? "on" : "OFF", g_vtf[1] ? "on" : g_cam ? "OFF" : "no camera");
		if (g_align && g_vtf[0] && g_vtf[1]) { /* the alignment's background arbiter reads the newest frame of each camera */
			g_lastk = malloc((size_t)W * H * 3);
			g_lastw = malloc((size_t)W * H * 3);
			if (!g_lastk || !g_lastw) { free(g_lastk); free(g_lastw); g_lastk = g_lastw = NULL; }
		}
	}
	if (g_fsk) { /* frame generation is set up whenever the engines run, so G can turn it on later */
		p_fg = build_program(S_FG, -1, -1);
		glProgramUniform1f(p_fg, 0, 1.0f);
		fg_setup(&g_fgk, t_rgb);
		if (g_fsw) fg_setup(&g_fgw, t_wrgb);
	}
	if (fg_test > 0) {
		if (!g_fsk || fgt_setup()) { fprintf(stderr, "kinect3d: --fg-test needs the FSR engines (cpus 8-11 for the third)\n"); return 1; }
		g_tsecs = fg_test;
		printf("kinect3d: fg-test: every second Kinect frame to a third engine, %.0f s of triples\n", g_tsecs);
	}
	if (g_cam) g_fw = flow_create(W, H, "webcam");
	glfwSetKeyCallback(win, on_key);
	glfwSetMouseButtonCallback(win, on_button);
	glfwSetCursorPosCallback(win, on_cursor);
	glfwSetScrollCallback(win, on_scroll);

	const uint8_t *d0, *v0;
	if (wait_first_frames(win, &d0, &v0)) { glfwTerminate(); kinect_close(); return 1; }
	freenect_registration reg = freenect_copy_registration(fdev);
	upload_registration(&reg);
	if (g_cam && !calib_dir && !track_dir && !face_dir && strcmp(calib_file ? calib_file : "", "none")) {
		const char *cf = calib_file ? calib_file : calib_default;
		if (!load_calib(cf)) {
			g_occl = 2;
			printf("kinect3d: webcam calibration %s: f %.1f %.1f c %.1f %.1f, k1 %.4f k2 %.4f, t %.1f %.1f %.1f mm\n", cf,
			       (double)g_cal.K[0], (double)g_cal.K[1], (double)g_cal.K[2], (double)g_cal.K[3], (double)g_cal.D[0],
			       (double)g_cal.D[1], (double)g_cal.t[0], (double)g_cal.t[1], (double)g_cal.t[2]);
		} else {
			printf("kinect3d: no webcam calibration (%s): occlusion gaps stay grey; run: make calibrate\n", cf);
		}
	}
	apply_calib();
	if (occl_start >= 0 && occl_start < (g_cal.ok ? 4 : 2)) g_occl = occl_start;
	if (g_cam && g_cal.ok) lock_setup();
	else g_lock = g_lock_measure = 0;
	printf("kinect3d: zero plane: reference distance %.4f mm, pixel size %.6f mm -> f %.2f px, IR-RGB %.4f cm, start_lines %u\n",
	       (double)reg.zero_plane_info.reference_distance, (double)reg.zero_plane_info.reference_pixel_size,
	       1.0 / (double)g_k, (double)reg.zero_plane_info.dcmos_rcmos_dist, (unsigned)reg.reg_pad_info.start_lines);
	upload_rgb(v0);
	g_dfilter = !do_selftest; /* the selftest compares against libfreenect on the raw depth */
	upload_depth(d0);
	{
		freenect_raw_tilt_state *ts;
		freenect_update_tilt_state(fdev);
		ts = freenect_get_tilt_state(fdev);
		tilt_deg = (int)lround(freenect_get_tilt_degs(ts));
	}

	if (do_selftest) {
		int f = selftest(win, (const uint16_t *)d0, v0, &reg);
		printf("selftest: %d FAIL\n", f);
		freenect_destroy_registration(&reg);
		glfwTerminate();
		kinect_close();
		return f;
	}

	cam_reset();
	cam.yaw = view_yaw;
	cam.pitch = view_pitch;
	enum { NQ = 8 };
	GLuint q[NQ];
	int q_method[NQ] = {0}, q_busy[NQ] = {0}, qi = 0;
	glCreateQueries(GL_TIME_ELAPSED, NQ, q);
	uint64_t *bns[2] = {NULL, NULL};
	int bn[2] = {0, 0}, bench_left = bench;
	if (bench > 0) {
		bns[0] = malloc(sizeof(uint64_t) * (size_t)bench);
		bns[1] = malloc(sizeof(uint64_t) * (size_t)bench);
		if (!bns[0] || !bns[1]) return 1;
		printf("kinect3d: bench: %d frames, paths A and B alternating every frame\n", bench);
	}
	double t_title = now_s(), gpu_ema[2] = {0, 0};
	unsigned nd_prev = atomic_load(&k_ndepth), nv_prev = atomic_load(&k_nrgb), frames = 0;
	unsigned fgk_prev = 0, fgw_prev = 0; /* frame generation: distinct images shown */
	unsigned nfk = 0, nfw = 0, nw_prev = 0;
	unsigned calib_snap = 0, sf_frames = 0, st_frames = 0, nfk_sf = 0, ndepth_seen = 0, nfk_fsr = 0;
	double td_prev = 0, d_period = 0;
	enum { ST_CAP = 200000 };
	static double st_v[4][ST_CAP], st_c[3][4][ST_CAP];
	unsigned st_n[4] = {0, 0, 0, 0};
	facetrack *ft = NULL;
	FILE *fc_k = NULL, *fc_w = NULL;
	unsigned nk_face = 0, nw_face = 0;
	int fc_phase = 0; /* 0 tracking, 1 snapshots, 2 done */
	double fc_t0 = 0;
	if (face_dir) {
		char p[1024];
		ft = ft_create(face_model ? face_model : face_model_default, W, H, 4);
		snprintf(p, sizeof p, "%s/face-kinect.txt", face_dir);
		fc_k = fopen(p, "w");
		snprintf(p, sizeof p, "%s/face-webcam.txt", face_dir);
		fc_w = fopen(p, "w");
		if (!ft || !fc_k || !fc_w || !g_cam) { fprintf(stderr, "kinect3d: --calib-face needs the webcam, the YuNet model and a writable %s\n", face_dir); return 1; }
		fprintf(fc_k, "# t_arrival_s then x y z (mm) of right eye, left eye, nose tip, right mouth, left mouth (nan = no depth), score\n");
		fprintf(fc_w, "# t_kernel_s then u v of right eye, left eye, nose tip, right mouth, left mouth, score\n");
		printf("kinect3d: FACE: look toward both cameras and move your head slowly for %.0f s: turn, tilt, lean in and out\n", face_secs);
	}
	static brightmask bm_k, bm_w;
	FILE *trk_k = NULL, *trk_w = NULL;
	unsigned nk_trk = 0, nw_trk = 0;
	int trk_phase = 0; /* 0 learning static lights, 1 tracking, 2 snapshots, 3 done */
	double trk_t0 = 0;
	if (track_dir) {
		char p[1024];
		snprintf(p, sizeof p, "%s/track-kinect.txt", track_dir);
		trk_k = fopen(p, "w");
		snprintf(p, sizeof p, "%s/track-webcam.txt", track_dir);
		trk_w = fopen(p, "w");
		if (!trk_k || !trk_w || !g_cam) { fprintf(stderr, "kinect3d: --calib-track needs the webcam and a writable %s\n", track_dir); return 1; }
		fprintf(trk_k, "# t_arrival_s x_mm y_mm z_mm rgb_x rgb_y pixels\n");
		fprintf(trk_w, "# t_kernel_s u v pixels\n");
		printf("kinect3d: TRACK: keep the light OFF for 1.5 s (learning static lights), then wave it for %.0f s\n", track_secs);
	}
	static const char *const OCCL_NAME[4] = {"off", "grey", "webcam fill", "webcam all"};

	while (!glfwWindowShouldClose(win)) {
		glfwPollEvents();
		int dslot = tb_acquire_slot(&tb_depth);
		const uint8_t *d = dslot < 0 ? NULL : tb_depth.buf[dslot];
		int vs = tb_acquire_slot(&tb_rgb);
		const double tloop = now_s();
		if (vs >= 0) {
			fg_new_frame(&g_fgk, k_rgb_t[vs], tloop); /* before the upload: cur -> prev */
			upload_rgb(tb_rgb.buf[vs]);
			g_t_rgb = k_rgb_t[vs];
			flow_push(g_fk, t_rgb, k_rgb_t[vs]);
			if (g_fsk) fsr_push(g_fsk, tb_rgb.buf[vs]);
			if (g_fst) fgt_frame(tb_rgb.buf[vs], nfk, k_rgb_t[vs]);
			if (g_vtf[0]) vtf_push(g_vtf[0], tb_rgb.buf[vs], k_rgb_t[vs]);
			if (g_lastk) memcpy(g_lastk, tb_rgb.buf[vs], (size_t)W * H * 3);
			nfk++;
		}
		double wt = 0;
		const uint8_t *wv = g_cam ? webcam_acquire(g_cam, &wt, NULL) : NULL;
		if (wv) {
			fg_new_frame(&g_fgw, wt, tloop);
			if (g_vtf[1]) vtf_push(g_vtf[1], wv, wt);
			if (g_lastw) memcpy(g_lastw, wv, (size_t)W * H * 3);
			glTextureSubImage2D(t_wrgb, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, wv);
			flow_push(g_fw, t_wrgb, wt);
			if (g_fsw) fsr_push(g_fsw, wv);
			if (g_flock && (g_lock || g_lock_measure) && flow_ready(g_fw)) {
				lock_step();
				if (lock_dump && nfw >= 90) { dump_lock(lock_dump); glfwSetWindowShouldClose(win, 1); lock_dump = NULL; }
			}
			nfw++;
		}
		uint32_t fseq[2];
		if (g_fsk && fsr_poll(g_fsk, g_fsbuf, g_rawbuf, fseq)) {
			glTextureSubImage2D(t_fsk, 0, 0, 0, 80, 60, GL_RGBA, GL_FLOAT, g_fsbuf);
			fg_field(&g_fgk, fseq);
			nfk_fsr++;
		}
		if (g_fsw && fsr_poll(g_fsw, g_fsbuf, g_rawbuf, fseq)) {
			glTextureSubImage2D(t_fsw, 0, 0, 0, 80, 60, GL_RGBA, GL_FLOAT, g_fsbuf);
			fg_field(&g_fgw, fseq);
		}
		for (int c = 0; c < 2; c++) /* face tracker results: tally states, the Kinect's into the room */
			if (g_vtf[c] && vtf_get(g_vtf[c], &g_fo[c])) {
				g_fstate[c][g_fo[c].state < 6 ? g_fo[c].state : 0]++;
				g_fres[c]++;
				if (!c) face_update();
				if (!g_t0_face && g_fo[c].faces) g_t0_face = now_s();
			}
		align_step();
		if (align_dump && g_lastk && g_lastw && nfk > 150 && nfw > 150) { /* --dump-align: one frame of each camera, the point map, the pose in use */
			char p[1024];
			int bad = al_bg_prepare(2);
			const void *src[3] = {g_lastk, g_lastw, g_bgpm};
			const size_t len[3] = {(size_t)W * H * 3, (size_t)W * H * 3, sizeof(float) * 4 * W * H};
			static const char *const name[3] = {"kinect.rgb", "webcam.rgb", "pmap.f32"};
			for (int k = 0; k < 3 && !bad; k++) {
				snprintf(p, sizeof p, "%s/%s", align_dump, name[k]);
				FILE *f = fopen(p, "wb");
				bad |= !f || fwrite(src[k], 1, len[k], f) != len[k];
				if (f) fclose(f);
			}
			snprintf(p, sizeof p, "%s/calib.txt", align_dump);
			bad |= save_calib(p);
			printf("dump-align: %s %s\n", align_dump, bad ? "FAILED" : "written (kinect.rgb webcam.rgb 640x480 RGB24, pmap.f32 640x480 RGBA32F, calib.txt)");
			align_dump = NULL;
			glfwSetWindowShouldClose(win, 1);
		}
		if (face_stats > 0 && g_t0_face && now_s() - g_t0_face >= face_stats) glfwSetWindowShouldClose(win, 1);
		if (g_fst) {
			fgt_poll();
			if (!g_t0_test && g_ft.triples) g_t0_test = now_s();
			if (g_t0_test && now_s() - g_t0_test >= g_tsecs) glfwSetWindowShouldClose(win, 1);
		}
		if (d) upload_depth(d); /* before the dumps: a snapshot pairs the newest depth with the newest pictures */
		if (d) { /* frame period, room rotation, velocity field */
			const double td = k_depth_t[dslot];
			g_t_depth = td;
			if (td_prev > 0 && td - td_prev > 0.015 && td - td_prev < 0.1) d_period = d_period > 0 ? 0.95 * d_period + 0.05 * (td - td_prev) : td - td_prev;
			td_prev = td;
			memcpy(g_room.Rprev, g_room.Rnow, sizeof g_room.Rnow);
			room_accel();
			if (ndepth_seen++ % 90 == 0) room_geometry((const uint16_t *)d);
			room_rotation(g_room.Rnow);
			if (!g_room.moving) memcpy(g_room.Rprev, g_room.Rnow, sizeof g_room.Rnow); /* still Kinect: no ego-rotation; accelerometer jitter would read as motion */
			const unsigned nflow = g_fsrc ? nfk_fsr : nfk;
			if ((g_fsrc ? nfk_fsr > 0 : flow_ready(g_fk)) && nflow != nfk_sf && d_period > 0) {
				sflow_step(d_period, g_fsrc ? t_fsk : flow_result(g_fk));
				nfk_sf = nflow;
				sf_frames++;
				if (stats_n > 0 && sf_frames > 30 && st_frames < (unsigned)stats_n) { /* noise floor: every cell, by depth band */
					for (int c = 0; c < 80 * 60; c++) {
						const float *v = g_cvb + 4 * c, *x = g_cpb + 4 * c;
						if (v[3] < 32 || x[3] < 32) continue;
						const int band = x[2] < 1000 ? 0 : x[2] < 2000 ? 1 : x[2] < 3000 ? 2 : 3;
						if (st_n[band] < ST_CAP) {
							st_v[band][st_n[band]] = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
							for (int k = 0; k < 3; k++) st_c[k][band][st_n[band]] = fabs(v[k]);
							st_n[band]++;
						}
					}
					if (++st_frames == (unsigned)stats_n) {
						static const char *band_name[4] = {"< 1 m", "1-2 m", "2-3 m", ">= 3 m"};
						printf("scene-stats: %u velocity fields, period %.2f ms; cells of 8x8 px with >= 32 valid pixels, |v| in mm/s\n",
						       st_frames, d_period * 1e3);
						for (int b = 0; b < 4; b++) {
							if (!st_n[b]) continue;
							qsort(st_v[b], st_n[b], sizeof(double), cmp_d);
							double cm[3];
							for (int k = 0; k < 3; k++) { qsort(st_c[k][b], st_n[b], sizeof(double), cmp_d); cm[k] = st_c[k][b][st_n[b] / 2]; }
							printf("scene-stats: %-6s %7u cells: |v| median %7.1f  p90 %7.1f; median |right| %6.1f |up| %6.1f |forward| %6.1f\n", band_name[b],
							       st_n[b], st_v[b][st_n[b] / 2], st_v[b][st_n[b] * 9 / 10], cm[0], cm[1], cm[2]);
						}
					}
				}
			}
		}
		if (do_analyze && d && atomic_load(&k_ndepth) > 60) {
			uint16_t *fd = malloc(sizeof(uint16_t) * NPIX);
			printf("analyze: RAW depth (as the Kinect sent it)\n");
			analyze_depth((const uint16_t *)d);
			if (fd) {
				glGetTextureImage(t_depth, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, sizeof(uint16_t) * NPIX, fd);
				printf("analyze: FILTERED depth (filter %s)\n", g_dfilter ? "on" : "off");
				analyze_depth(fd);
				free(fd);
			}
			do_analyze = 0;
		}
		if (face_dir && nfw >= 60 && fc_phase < 2) { /* past the webcam's ~2 s start-up */
			double tn = now_s();
			if (!fc_t0) fc_t0 = tn;
			if (fc_phase == 0) {
				face5 f;
				float X[15];
				if (d) compute_pmap();
				if (vs >= 0 && ft_detect(ft, tb_rgb.buf[vs], &f) > 0 && face_3d(&f, X) >= 3) {
					fprintf(fc_k, "%.6f", k_rgb_t[vs]);
					for (int k = 0; k < 15; k++) fprintf(fc_k, " %.2f", (double)X[k]);
					fprintf(fc_k, " %.3f\n", (double)f.score);
					nk_face++;
				}
				if (wv && ft_detect(ft, wv, &f) > 0) {
					fprintf(fc_w, "%.6f", wt);
					for (int k = 0; k < 10; k++) fprintf(fc_w, " %.3f", (double)f.lm[k]);
					fprintf(fc_w, " %.3f\n", (double)f.score);
					nw_face++;
				}
				if (tn - fc_t0 > face_secs) {
					fclose(fc_k);
					fclose(fc_w);
					printf("kinect3d: FACE: %u Kinect and %u webcam face samples; 3 snapshots for the colour fit\n", nk_face, nw_face);
					fc_phase = 1;
					calib_snap = 0;
				}
			} else if (d && nfw >= 15 * calib_snap) {
				dump_calib(face_dir, (int)calib_snap);
				if (++calib_snap == 3u) { fc_phase = 2; glfwSetWindowShouldClose(win, 1); }
			}
		}
		if (track_dir && nfw >= 60 && trk_phase < 3) { /* past the webcam's ~2 s start-up */
			double tn = now_s();
			if (!trk_t0) trk_t0 = tn;
			int learn = tn - trk_t0 < 1.5;
			if (!learn && trk_phase == 0) {
				bm_finish(&bm_k);
				bm_finish(&bm_w);
				trk_phase = 1;
				printf("kinect3d: TRACK: static lights learned (%u / %u frames); wave the light now\n", bm_k.frames, bm_w.frames);
			}
			if (trk_phase <= 1) {
				float cx, cy, X[3];
				unsigned n;
				if (d) compute_pmap();
				if (vs >= 0 && bright_blob(tb_rgb.buf[vs], &bm_k, learn, &cx, &cy, &n) && kinect_point(cx, cy, X)) {
					fprintf(trk_k, "%.6f %.2f %.2f %.2f %.2f %.2f %u\n", k_rgb_t[vs], (double)X[0], (double)X[1], (double)X[2],
						(double)cx, (double)cy, n);
					nk_trk++;
				}
				if (wv && bright_blob(wv, &bm_w, learn, &cx, &cy, &n)) {
					fprintf(trk_w, "%.6f %.3f %.3f %u\n", wt, (double)cx, (double)cy, n);
					nw_trk++;
				}
				if (trk_phase == 1 && tn - trk_t0 > 1.5 + track_secs) {
					fclose(trk_k);
					fclose(trk_w);
					printf("kinect3d: TRACK: %u Kinect and %u webcam detections; 3 snapshots for the colour fit\n", nk_trk, nw_trk);
					trk_phase = 2;
					calib_snap = 0;
				}
			} else if (trk_phase == 2 && d && nfw >= 15 * calib_snap) {
				dump_calib(track_dir, (int)calib_snap);
				if (++calib_snap == 3u) { trk_phase = 3; glfwSetWindowShouldClose(win, 1); }
			}
		}
		if (calib_dir && g_cam && nfw >= 60 + 15 * calib_snap && d) { /* 5 snapshots, 15 webcam frames apart */
			if (dump_calib(calib_dir, (int)calib_snap)) glfwSetWindowShouldClose(win, 1);
			if (++calib_snap == 5u) glfwSetWindowShouldClose(win, 1);
		}
		if (dump_dir && nfk >= 60 && (!g_cam || nfw >= 60)) {
			int e = dump_flow(dump_dir, g_fk, "kinect");
			if (g_cam) e |= dump_flow(dump_dir, g_fw, "webcam");
			printf("dump-flow: %s\n", e ? "FAILED" : "written");
			glfwSetWindowShouldClose(win, 1);
		}

		int fbw, fbh;
		glfwGetFramebufferSize(win, &fbw, &fbh);
		glViewport(0, 0, fbw, fbh);
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.07f, 0.07f, 0.08f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		mat4 mvp = cam_mvp(fbw, fbh);
		int m = bench_left > 0 ? bench_left & 1 : g_method;

		if (q_busy[qi]) { /* eight frames old: available without a stall */
			GLuint64 ns = 0;
			glGetQueryObjectui64v(q[qi], GL_QUERY_RESULT, &ns);
			int qm = q_method[qi];
			gpu_ema[qm] = gpu_ema[qm] ? 0.9 * gpu_ema[qm] + 0.1 * (double)ns : (double)ns;
			if (q_busy[qi] == 2 && bn[qm] < bench) bns[qm][bn[qm]++] = ns;
		}
		/* frame generation: a field still due within 4 ms of its frame's arrival is waited for, so the generated
		 * image makes this refresh instead of the next (field 1.8-2.2 ms, refresh 8.33 ms at 120 Hz) */
		if (g_fgk.gen && g_fg && g_fgk.phase == 0 && fsr_wait(g_fsk, g_fgk.t_arr[g_fgk.seq % 4] + 0.004) && fsr_poll(g_fsk, g_fsbuf, g_rawbuf, fseq)) {
			glTextureSubImage2D(t_fsk, 0, 0, 0, 80, 60, GL_RGBA, GL_FLOAT, g_fsbuf);
			fg_field(&g_fgk, fseq);
			nfk_fsr++;
		}
		if (g_fgw.gen && g_fg && g_fgw.phase == 0 && fsr_wait(g_fsw, g_fgw.t_arr[g_fgw.seq % 4] + 0.004) && fsr_poll(g_fsw, g_fsbuf, g_rawbuf, fseq)) {
			glTextureSubImage2D(t_fsw, 0, 0, 0, 80, 60, GL_RGBA, GL_FLOAT, g_fsbuf);
			fg_field(&g_fgw, fseq);
		}
		const double tdraw = now_s(); /* frame generation: what each camera shows at this refresh */
		g_show_k = g_fgk.gen ? fg_display(&g_fgk, tdraw) : t_rgb;
		g_show_w = g_fgw.gen ? fg_display(&g_fgw, tdraw) : t_wrgb;
		glBindTextureUnit(4, g_show_k);
		{ /* the geometry at the instant of the Kinect colour on screen (prev, the generated midpoint, or cur) */
			const double tcol = g_fgk.gen && g_fgk.t_cap_prev > 0 ? (g_fgk.show == 0 ? g_fgk.t_cap_prev : g_fgk.show == 1 ? 0.5 * (g_fgk.t_cap_prev + g_fgk.t_cap_last) : g_fgk.t_cap_last) : g_t_rgb;
			const double ts = g_geo && g_t_depth > 0 && tcol > 0 && sf_frames ? tcol - g_t_depth : 0.0;
			g_tshift = (float)(ts > 0.05 ? 0.05 : ts < -0.05 ? -0.05 : ts);
			if (g_tshift != 0.0f) { g_tshift_sum += fabs(g_tshift); g_tshift_n++; }
		}
		glBindTextureUnit(12, g_show_w);
		glBeginQuery(GL_TIME_ELAPSED, q[qi]);
		draw_mesh(m, &mvp);
		glEndQuery(GL_TIME_ELAPSED);
		if (g_arrows && sf_frames) draw_arrows(&mvp);
		if (g_pip) draw_pip(fbw, fbh);
		glBindTextureUnit(4, t_rgb); /* the analysis passes read the real frames */
		glBindTextureUnit(12, t_wrgb);
		q_method[qi] = m;
		q_busy[qi] = bench_left > 0 ? 2 : 1;
		qi = (qi + 1) % NQ;
		if (bench_left > 0 && --bench_left == 0) glfwSetWindowShouldClose(win, 1);

		glfwSwapBuffers(win);
		frames++;
		double t = now_s();
		if (t - t_title >= 0.5) {
			unsigned nd = atomic_load(&k_ndepth), nv = atomic_load(&k_nrgb), nw = 0, wf = 0, wg = 0;
			double wdec = 0;
			if (g_cam) webcam_stats(g_cam, &nw, &wf, &wg, &wdec);
			char title[768];
			snprintf(title, sizeof title,
				 "kinect3d | depth %.1f  rgb %.1f  webcam %.1f fps | render %.0f fps | mesh %c %.3f ms | flow K %.2f W %.2f ms | "
				 "jpeg %.2f ms | cull %.1f %% | occl %s | depth filter %s",
				 (nd - nd_prev) / (t - t_title), (nv - nv_prev) / (t - t_title), (nw - nw_prev) / (t - t_title),
				 frames / (t - t_title), g_method ? 'B' : 'A', gpu_ema[g_method] * 1e-6, flow_gpu_ms(g_fk),
				 g_fw ? flow_gpu_ms(g_fw) : 0.0, wdec, g_cull * 100.0, OCCL_NAME[g_occl], g_dfilter ? "on" : "off");
			if (face_dir && fc_phase < 2)
				snprintf(title, sizeof title, "kinect3d FACE | %s | Kinect %u  webcam %u face samples | %.0f s left",
					 fc_phase == 0 ? "MOVE YOUR HEAD SLOWLY: turn, tilt, lean in and out" : "snapshots", nk_face, nw_face,
					 fc_t0 ? face_secs - (t - fc_t0) : face_secs);
			if (track_dir && trk_phase < 3)
				snprintf(title, sizeof title, "kinect3d TRACK | %s | Kinect %u  webcam %u detections | %.0f s left",
					 trk_phase == 0 ? "keep the light OFF: learning static lights" : trk_phase == 1 ? "WAVE THE LIGHT" : "snapshots",
					 nk_trk, nw_trk, trk_t0 ? 1.5 + track_secs - (t - trk_t0) : 0.0);
			if (g_vtf[0] || g_vtf[1]) { /* face tracker */
				static const char *const ST[6] = {"none", "new", "observed", "unobserved", "lost", "deselected"};
				char fs[260];
				snprintf(fs, sizeof fs, " | face K %s %u feat W %s %u feat", ST[g_fo[0].state % 6], g_fo[0].members, ST[g_fo[1].state % 6], g_fo[1].members);
				strncat(title, fs, sizeof title - strlen(title) - 1);
				if (g_facep.have && t - g_facep.t_last < 0.5) {
					snprintf(fs, sizeof fs, " at (%.2f, %.2f, %.2f) m v (%.2f, %.2f, %.2f) m/s", g_facep.pos[0] * 1e-3, g_facep.pos[1] * 1e-3, g_facep.pos[2] * 1e-3,
						 g_facep.vel[0] * 1e-3, g_facep.vel[1] * 1e-3, g_facep.vel[2] * 1e-3);
					strncat(title, fs, sizeof title - strlen(title) - 1);
				}
				if (g_align && g_al.solves) {
					snprintf(fs, sizeof fs, " | align %d corr: test %.1f -> %.1f px%s", g_al.n, g_al.old_med, g_al.new_med, g_al.applied ? " APPLIED" : "");
					strncat(title, fs, sizeof title - strlen(title) - 1);
				}
				if (face_stats > 0)
					printf("face: K %s box (%.0f, %.0f) %.0fx%.0f %u/%u feat iou %.2f | W %s box (%.0f, %.0f) %.0fx%.0f %u/%u feat iou %.2f | room %.0f %.0f %.0f mm v %.0f %.0f %.0f mm/s\n",
					       ST[g_fo[0].state % 6], g_fo[0].cx, g_fo[0].cy, g_fo[0].w, g_fo[0].h, g_fo[0].members, g_fo[0].nfeat, g_fo[0].iou,
					       ST[g_fo[1].state % 6], g_fo[1].cx, g_fo[1].cy, g_fo[1].w, g_fo[1].h, g_fo[1].members, g_fo[1].nfeat, g_fo[1].iou,
					       g_facep.pos[0], g_facep.pos[1], g_facep.pos[2], g_facep.vel[0], g_facep.vel[1], g_facep.vel[2]);
			}
			if (g_fgk.gen) { /* frame generation: distinct images on screen per second, per camera */
				char fgs[160];
				const double sk = (g_fgk.shown - fgk_prev) / (t - t_title), sw = (g_fgw.shown - fgw_prev) / (t - t_title);
				snprintf(fgs, sizeof fgs, " | frame gen %s: shown K %.1f W %.1f fps", g_fg ? "on" : "off", sk, sw);
				strncat(title, fgs, sizeof title - strlen(title) - 1);
				if (stats_n > 0 || fg_test > 0) printf("fg: %s shown K %.1f W %.1f fps (camera %.1f / %.1f, render %.1f) generated K %u W %u\n", g_fg ? "on" : "off",
								      sk, sw, (nv - nv_prev) / (t - t_title), (nw - nw_prev) / (t - t_title), frames / (t - t_title), g_fgk.made, g_fgw.made);
			}
			if (sf_frames) { /* room frame: x right (along a wall), y up (gravity), z forward; origin = Kinect */
				char rm[300];
				snprintf(rm, sizeof rm, " | room: gravity vs ceiling %.1f deg, ceiling +%.2f m, side walls %.2f / +%.2f m | moving %u cells at (%.2f, %.2f, %.2f) m, v (%.2f, %.2f, %.2f) m/s",
					 g_room.ceil_angle, g_room.ceil_mm * 1e-3, g_room.wall_left * 1e-3, g_room.wall_right * 1e-3, g_mov.n,
					 g_mov.pos[0] * 1e-3, g_mov.pos[1] * 1e-3, g_mov.pos[2] * 1e-3, g_mov.vel[0] * 1e-3, g_mov.vel[1] * 1e-3, g_mov.vel[2] * 1e-3);
				strncat(title, rm, sizeof title - strlen(title) - 1);
				printf("room: up/ceiling %.2f deg (signs x %+d z %+d, %u ceiling, %u wall pts) ceiling %.0f mm walls %.0f %.0f ahead %.0f | moving %u at %.0f %.0f %.0f mm v %.0f %.0f %.0f mm/s | period %.2f ms\n",
				       g_room.ceil_angle, g_room.sx, g_room.sz, g_room.ceil_n, g_room.wall_n, g_room.ceil_mm, g_room.wall_left, g_room.wall_right,
				       g_room.wall_ahead, g_mov.n, g_mov.pos[0], g_mov.pos[1], g_mov.pos[2], g_mov.vel[0], g_mov.vel[1], g_mov.vel[2], d_period * 1e3);
			}
			if (g_lock || g_lock_measure) {
				char lk[160];
				snprintf(lk, sizeof lk, " | LOCK %s: %u px, mean residual %.3f px, step %.3f deg %.2f mm, %u updates", g_lock ? "on" : "measure", g_lk.n,
					 g_lk.mean_r, g_lk.step_deg, g_lk.step_mm, g_lk.updates);
				strncat(title, lk, sizeof title - strlen(title) - 1);
				printf("lock: %s n %u mean_r %.3f px step %.3f deg %.2f mm updates %u t %.1f %.1f %.1f\n", g_lock ? "on" : "measure", g_lk.n,
				       g_lk.mean_r, g_lk.step_deg, g_lk.step_mm, g_lk.updates, (double)g_cal.t[0], (double)g_cal.t[1], (double)g_cal.t[2]);
			}
			glfwSetWindowTitle(win, title);
			nd_prev = nd; nv_prev = nv; nw_prev = nw; frames = 0; t_title = t;
			fgk_prev = g_fgk.shown; fgw_prev = g_fgw.shown;
		}
	}

	if (bench > 0) {
		glFinish();
		for (int k = 0; k < NQ; k++) /* drain the queries still in flight */
			if (q_busy[k] == 2 && bn[q_method[k]] < bench) {
				GLuint64 ns = 0;
				glGetQueryObjectui64v(q[k], GL_QUERY_RESULT, &ns);
				bns[q_method[k]][bn[q_method[k]]++] = ns;
			}
		double med[2] = {0, 0};
		for (int k = 0; k < 2; k++) {
			if (!bn[k]) continue;
			size_t n = (size_t)bn[k], i_med = n / 2, i_p90 = n * 9 / 10; /* integer index positions */
			qsort(bns[k], n, sizeof(uint64_t), cmp_u64);
			med[k] = (double)bns[k][i_med];
			printf("bench path %c: %d draws, GPU min %.3f  median %.3f  p90 %.3f  max %.3f ms\n", k ? 'B' : 'A', bn[k],
			       (double)bns[k][0] * 1e-6, med[k] * 1e-6, (double)bns[k][i_p90] * 1e-6, (double)bns[k][n - 1] * 1e-6);
		}
		if (med[0] > 0 && med[1] > 0)
			printf("bench verdict: path %c faster by %.3fx (median)\n", med[1] <= med[0] ? 'B' : 'A',
			       med[1] <= med[0] ? med[0] / med[1] : med[1] / med[0]);
		printf("bench GL errors %u, buffer protocol mismatches %u\n", gl_errors, atomic_load(&k_bufmiss));
		free(bns[0]); free(bns[1]);
	}
	freenect_destroy_registration(&reg);
	if (g_cam) {
		unsigned nw, wf, wg;
		double wdec;
		webcam_stats(g_cam, &nw, &wf, &wg, &wdec);
		printf("kinect3d: webcam %u frames, %u decode failures, %u decoded despite a warning, %u dropped (sequence gaps), JPEG decode %.3f ms mean%s%s\n",
		       nw, wf, webcam_warnings(g_cam), wg, wdec, *webcam_error(g_cam) ? "; first: " : "", webcam_error(g_cam));
		webcam_close(g_cam);
	}
	if (ft) ft_destroy(ft);
	if (g_align && g_vtf[0] && g_vtf[1]) {
		printf("kinect3d: face alignment: detections still K %u W %u, moving %u, pairs %u (unpaired in time %u), kept %u, %d correspondences, "
		       "head travel %.0f mm, solves %u, applied %u, cleared by the motor %u\n", g_al.still_k, g_al.still_w, g_al.unstill, g_al.pairs, g_al.untimed, g_al.kept, g_al.n,
		       g_al.spread, g_al.solves, g_al.applied, g_al.resets);
		if (g_al.applied) {
			char ap[512];
			snprintf(ap, sizeof ap, "%s/kinect360/calib/webcam-aligned.txt", getenv("HOME") ? getenv("HOME") : ".");
			printf("kinect3d: aligned pose %s %s (the next start loads it; calib/webcam.txt untouched)\n", save_calib(ap) ? "NOT written to" : "written to", ap);
		}
	}
	for (int c = 0; c < 2; c++) { /* face tracker report */
		if (!g_vtf[c]) continue;
		const vtf_out *o = &g_fo[c];
		const unsigned *s = g_fstate[c];
		const double r = g_fres[c] ? 100.0 / g_fres[c] : 0.0;
		printf("kinect3d: face tracker %s: %u results: observed %.1f %%, unobserved %.1f %%, new %.1f %%, lost %.1f %%, none %.1f %%; "
		       "tracker %.3f ms mean %.3f max; YuNet %.2f ms, %llu runs, %llu faces; selects %llu (after lost %llu, drift %llu); "
		       "box on the face (IoU >= 0.5 at a detection) %llu of %llu = %.1f %%, centre inside the face %.1f %%, centre error %.3f face widths (mean)\n",
		       c ? "webcam" : "Kinect", g_fres[c], s[VTF_OBSERVED] * r, s[VTF_UNOBSERVED] * r, s[VTF_NEW] * r, s[VTF_LOST] * r, s[VTF_NONE] * r,
		       o->track_ms, o->track_ms_max, o->face_ms, (unsigned long long)o->detections, (unsigned long long)o->faces,
		       (unsigned long long)o->selects, (unsigned long long)o->reselect_lost, (unsigned long long)o->reselect_drift,
		       (unsigned long long)o->on_face, (unsigned long long)o->judged, o->judged ? 100.0 * (double)o->on_face / (double)o->judged : 0.0,
		       o->judged ? 100.0 * (double)o->centre_in / (double)o->judged : 0.0, o->judged ? o->centre_err / (double)o->judged : 0.0);
		vtf_close(g_vtf[c]);
	}
	if (g_vtf[0]) printf("kinect3d: face in the room: %u points, %u without depth under the box\n", g_facep.points, g_facep.misses);
	if (g_fst) fgt_report();
	if (g_tshift_n) printf("kinect3d: geometry at the colour's instant: mean |shift| %.2f ms over %u draws\n", 1e3 * g_tshift_sum / g_tshift_n, g_tshift_n);
	if (g_fgk.gen) printf("kinect3d: frame generation: Kinect %u generated, %u images shown, %u fields late, %u periods without one; webcam %u, %u, %u, %u\n", g_fgk.made, g_fgk.shown, g_fgk.late, g_fgk.nofield, g_fgw.made, g_fgw.shown, g_fgw.late, g_fgw.nofield);
	if (g_fgk.lat_n) printf("kinect3d: frame generation latency, loop arrival to first refresh of a real frame: Kinect mean %.2f ms max %.2f (%u), webcam mean %.2f max %.2f (%u)\n", 1e3 * g_fgk.lat_sum / g_fgk.lat_n, 1e3 * g_fgk.lat_max, g_fgk.lat_n, g_fgw.lat_n ? 1e3 * g_fgw.lat_sum / g_fgw.lat_n : 0.0, 1e3 * g_fgw.lat_max, g_fgw.lat_n);
	for (int k = 0; k < 3; k++) {
		static const char *const NAME[3] = {"Kinect", "webcam", "fg-test"};
		fsrcam *fc = k == 0 ? g_fsk : k == 1 ? g_fsw : g_fst;
		if (!fc) continue;
		unsigned nf, ns, nsc;
		double ems;
		fsr_stats(fc, &nf, &ns, &nsc, &ems);
		printf("kinect3d: FSR flow %s: %u fields, %u frames skipped (engine busy), %u scene changes, %.3f ms a field\n", NAME[k], nf, ns, nsc, ems);
		unsigned rf, kw, st;
		double rms;
		fsr_refine_stats(fc, &rf, &kw, &st, &rms);
		printf("kinect3d: FSR flow %s: VisionTrack refinement: %u blocks refined, %u moving kept whole, %u still; %.3f ms a field\n", NAME[k], rf, kw, st, rms);
		fsr_close(fc);
	}
	if (g_lock && g_lk.updates) printf("kinect3d: locked pose %s out/lock-pose.txt\n", save_calib("out/lock-pose.txt") ? "NOT written to" : "written to");
	glfwDestroyWindow(win);
	glfwTerminate();
	kinect_close();
	return 0;
}
