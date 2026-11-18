// kcalib.cpp: webcam <-> Kinect calibration from `kinect3d --calib-dump DIR` snapshots.
// Correspondences: SIFT on the Kinect RGB and webcam pictures, Lowe ratio 0.75 + mutual best match;
// the 3D point of a Kinect keypoint comes from the RGB-space point map (exact inverse registration).
// Solve: RANSAC PnP over a focal scan 300..1300 px, then calibrateCamera on the inliers (fx = fy,
// principal point, k1, k2), inliers re-selected twice at 2 px. Judged by physics first: the webcam
// must sit above the Kinect (the user's description), a few cm away, nearly parallel.
// Colour: per channel, kinect = gain * webcam + offset, trimmed least squares over projected points.
// Build: see Makefile target out/kcalib.  Usage: kcalib DIR OUT.txt
#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int W = 640, H = 480;

struct Snap {
	cv::Mat k, w;          // BGR pictures
	std::vector<float> pm; // RGBA32F point map, Kinect RGB space
};

static bool point_at(const std::vector<float> &pm, float x, float y, cv::Point3f &X)
{
	const int cx = (int)std::lround(x), cy = (int)std::lround(y);
	float best = 1e9f;
	bool ok = false;
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++) {
			const int px = cx + dx, py = cy + dy;
			if (px < 0 || py < 0 || px >= W || py >= H) continue;
			const float *p = &pm[4 * ((size_t)py * W + px)];
			const float d2 = (float)((px - x) * (px - x) + (py - y) * (py - y));
			if (p[3] > 0.5f && d2 < best) { best = d2; X = cv::Point3f(p[0], p[1], p[2]); ok = true; }
		}
	return ok;
}

static double median(std::vector<double> v)
{
	if (v.empty()) return NAN;
	std::nth_element(v.begin(), v.begin() + (long)(v.size() / 2), v.end());
	return v[v.size() / 2];
}

int main(int argc, char **argv)
{
	std::setvbuf(stdout, nullptr, _IOLBF, 0); /* an abort must not swallow the diagnostics */
	if (argc < 3) { std::fprintf(stderr, "usage: %s DIR OUT.txt [--focal F]\n", argv[0]); return 64; }
	const double f_fixed = argc >= 5 && !std::strcmp(argv[3], "--focal") ? std::atof(argv[4]) : 0.0;
	const std::string dir = argv[1];
	std::vector<Snap> snaps;
	for (int i = 0; i < 16; i++) {
		Snap s;
		s.k = cv::imread(dir + "/snap" + std::to_string(i) + "-kinect.ppm", cv::IMREAD_COLOR);
		s.w = cv::imread(dir + "/snap" + std::to_string(i) + "-webcam.ppm", cv::IMREAD_COLOR);
		std::ifstream in(dir + "/snap" + std::to_string(i) + "-points.f32", std::ios::binary);
		s.pm.resize((size_t)W * H * 4);
		if (s.k.empty() || s.w.empty() || !in.read(reinterpret_cast<char *>(s.pm.data()), (std::streamsize)(s.pm.size() * 4))) break;
		snaps.push_back(std::move(s));
	}
	if (snaps.empty()) { std::fprintf(stderr, "no snapshots in %s\n", dir.c_str()); return 1; }

	// track modes: one moving light (kinect3d --calib-track, 1 point per sample) or the face
	// (kinect3d --calib-face, 5 landmarks per sample: eyes, nose tip, mouth corners). Kinect 3D vs webcam
	// 2D of the same physical points; the two clocks (Kinect callback arrival, webcam kernel timestamp)
	// differ by an unknown latency, solved here. NaN = landmark without depth.
	enum { MAXP = 5 };
	struct KS { double t; cv::Point3f X[MAXP]; };
	struct WS { double t; cv::Point2f p[MAXP]; };
	std::vector<KS> ks;
	std::vector<WS> ws;
	int np = 0;
	const char *mode = "";
	auto load = [&](const char *kf, const char *wf, int n) {
		std::ifstream a(dir + "/" + kf), b(dir + "/" + wf);
		std::string line;
		while (std::getline(a, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream s(line);
			KS k;
			s >> k.t;
			for (int i = 0; i < n; i++) {
				std::string x, y, z; /* "nan" read as text: istream >> double rejects it */
				s >> x >> y >> z;
				k.X[i] = cv::Point3f(std::strtof(x.c_str(), nullptr), std::strtof(y.c_str(), nullptr), std::strtof(z.c_str(), nullptr));
			}
			if (s) ks.push_back(k);
		}
		while (std::getline(b, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream s(line);
			WS w;
			s >> w.t;
			for (int i = 0; i < n; i++) s >> w.p[i].x >> w.p[i].y;
			if (s) ws.push_back(w);
		}
	};
	load("face-kinect.txt", "face-webcam.txt", 5);
	if (ks.size() >= 30 && ws.size() >= 30) { np = 5; mode = "face"; }
	else {
		ks.clear();
		ws.clear();
		load("track-kinect.txt", "track-webcam.txt", 1);
		if (ks.size() >= 30 && ws.size() >= 30) { np = 1; mode = "light"; }
	}
	const bool track = np > 0;
	const bool face = np == 5;
	if (!track && (!ks.empty() || !ws.empty())) {
		std::printf("FAIL track: %zu Kinect and %zu webcam samples (need >= 30 each)\n", ks.size(), ws.size());
		return 1;
	}
	auto pairs = [&](double dt, std::vector<cv::Point3f> &O, std::vector<cv::Point2f> &I) {
		O.clear();
		I.clear();
		size_t j = 0;
		for (const KS &k : ks) {
			const double tq = k.t + dt;
			while (j + 1 < ws.size() && ws[j + 1].t <= tq) j++;
			if (j + 1 >= ws.size() || ws[j].t > tq) continue;
			const double span = ws[j + 1].t - ws[j].t;
			if (span <= 0 || span > 0.08) continue; /* no interpolation across missed detections */
			const float a = (float)((tq - ws[j].t) / span);
			for (int i = 0; i < np; i++) {
				if (std::isnan(k.X[i].z)) continue;
				O.push_back(k.X[i]);
				I.push_back(ws[j].p[i] * (1.0f - a) + ws[j + 1].p[i] * a);
			}
		}
	};
	auto inliers_at = [&](double dt, int f, float px, int iters) {
		std::vector<cv::Point3f> O;
		std::vector<cv::Point2f> I;
		pairs(dt, O, I);
		if (O.size() < 12) return (size_t)0;
		cv::Mat K(cv::Matx33d((double)f, 0, 319.5, 0, (double)f, 239.5, 0, 0, 1)), rv, tv;
		std::vector<int> inl;
		return cv::solvePnPRansac(O, I, K, cv::noArray(), rv, tv, false, iters, px, 0.99, inl) ? inl.size() : (size_t)0;
	};

	std::vector<cv::Point3f> obj;
	std::vector<cv::Point2f> img;
	double best_dt = 0;
	if (track) {
		std::printf("%s track: %zu Kinect and %zu webcam samples over %.1f s, %d point(s) each\n", mode, ks.size(), ws.size(), ks.back().t - ks.front().t, np);
		size_t bn = 0;
		int f0 = 0;
		for (int ms = -200; ms <= 200; ms += 10)
			for (int f = 400; f <= 1300; f += 100) {
				const size_t n = inliers_at(ms * 1e-3, f, face ? 8.0f : 6.0f, 500);
				if (n > bn) { bn = n; best_dt = ms * 1e-3; f0 = f; }
			}
		const double c0 = best_dt;
		bn = 0;
		for (int us = -15; us <= 15; us++) {
			const size_t n = inliers_at(c0 + us * 1e-3, f0, face ? 6.0f : 4.0f, 1000);
			if (n > bn) { bn = n; best_dt = c0 + us * 1e-3; }
		}
		pairs(best_dt, obj, img);
		std::printf("track: time offset %+.1f ms (webcam kernel time = Kinect arrival time %+.1f ms), %zu pairs, coarse f %d px, %zu inliers at 4 px\n",
			    best_dt * 1e3, best_dt * 1e3, obj.size(), f0, bn);
	}
	auto sift = cv::SIFT::create(4000);
	cv::BFMatcher bf(cv::NORM_L2);
	for (size_t si = 0; !track && si < snaps.size(); si++) {
		cv::Mat kg, wg, kd, wd;
		std::vector<cv::KeyPoint> kk, wk;
		cv::cvtColor(snaps[si].k, kg, cv::COLOR_BGR2GRAY);
		cv::cvtColor(snaps[si].w, wg, cv::COLOR_BGR2GRAY);
		sift->detectAndCompute(kg, cv::noArray(), kk, kd);
		sift->detectAndCompute(wg, cv::noArray(), wk, wd);
		std::vector<std::vector<cv::DMatch>> fw, bw;
		bf.knnMatch(kd, wd, fw, 2);
		bf.knnMatch(wd, kd, bw, 2);
		unsigned ratio = 0, mutual = 0, with3d = 0;
		for (const auto &m : fw) {
			if (m.size() < 2 || m[0].distance > 0.75f * m[1].distance) continue;
			ratio++;
			if (bw[(size_t)m[0].trainIdx].empty() || bw[(size_t)m[0].trainIdx][0].trainIdx != m[0].queryIdx) continue;
			mutual++;
			cv::Point3f X;
			const cv::Point2f kp = kk[(size_t)m[0].queryIdx].pt;
			if (!point_at(snaps[si].pm, kp.x, kp.y, X)) continue;
			with3d++;
			obj.push_back(X);
			img.push_back(wk[(size_t)m[0].trainIdx].pt);
		}
		std::printf("snap %zu: SIFT %zu kinect / %zu webcam keypoints, ratio %u, mutual %u, with a 3D point %u\n", si, kk.size(),
			    wk.size(), ratio, mutual, with3d);
	}
	if (obj.size() < 30) { std::printf("FAIL too few correspondences (%zu)\n", obj.size()); return 1; }

	// focal scan with RANSAC PnP
	int best_f = 0;
	size_t best_n = 0;
	std::vector<int> best_inl;
	for (int f = 300; f <= 1300; f += 10) {
		cv::Mat K(cv::Matx33d((double)f, 0, 319.5, 0, (double)f, 239.5, 0, 0, 1)), rv, tv;
		std::vector<int> inl;
		if (!cv::solvePnPRansac(obj, img, K, cv::noArray(), rv, tv, false, 2000, face ? 6.0f : track ? 4.0f : 3.0f, 0.999, inl)) continue;
		if (inl.size() > best_n) { best_n = inl.size(); best_f = f; best_inl = inl; }
	}
	std::printf("focal scan 300..1300: best f %d px with %zu of %zu inliers at %d px\n", best_f, best_n, obj.size(), face ? 6 : track ? 4 : 3);
	if (best_n < 30) { std::printf("FAIL focal scan found no consistent pose\n"); return 1; }

	cv::Mat K(cv::Matx33d((double)best_f, 0, 319.5, 0, (double)best_f, 239.5, 0, 0, 1)), D = cv::Mat::zeros(5, 1, CV_64F);
	std::vector<cv::Mat> rvs(1), tvs(1);
	std::vector<int> inl = best_inl;
	double rms = 0;
	const double sel = face ? 4.0 : track ? 3.0 : 2.0;
	if (track) {
		// A face (or a hand-held light) spans a small volume: focal length and distance nearly trade off
		// (weak perspective), and freeing f in calibrateCamera diverged (face run 13:39: rms 3.96 -> 23.1 px,
		// then a negative focal length). So: profile f, keep the best, solve the pose only at that f.
		auto pose_at = [&](double f, cv::Mat &rv, cv::Mat &tv, std::vector<int> &in3, double &med) {
			cv::Mat Kf(cv::Matx33d(f, 0, 319.5, 0, f, 239.5, 0, 0, 1));
			std::vector<int> r;
			if (!cv::solvePnPRansac(obj, img, Kf, cv::noArray(), rv, tv, false, 1000, face ? 6.0f : 4.0f, 0.999, r) || r.size() < 12) return false;
			for (int round = 0; round < 2; round++) {
				std::vector<cv::Point3f> O;
				std::vector<cv::Point2f> I;
				for (int i : r) { O.push_back(obj[(size_t)i]); I.push_back(img[(size_t)i]); }
				cv::solvePnPRefineLM(O, I, Kf, cv::noArray(), rv, tv);
				std::vector<cv::Point2f> pr;
				cv::projectPoints(obj, rv, tv, Kf, cv::noArray(), pr);
				r.clear();
				std::vector<double> e;
				for (size_t i = 0; i < obj.size(); i++) {
					const double d = cv::norm(pr[i] - img[i]);
					if (d < sel) { r.push_back((int)i); e.push_back(d); }
				}
				med = median(e);
				if (r.size() < 12) return false;
			}
			in3 = r;
			return true;
		};
		// Focal length from the face's apparent size against its depth (face mode). For a short 3D vector E
		// at a point of depth Z in the webcam frame, its image length is f * |E_xy - x_n E_z| / Z (x_n = the
		// point's normalised image position). With E = left eye - right eye from the Kinect and the eye
		// spacing s measured in the webcam: y = |E_xy - x_n E_z| / s = Z / f = (M_z + t_z) / f, where M_z is
		// the eyes' midpoint rotated into the webcam frame (translation excluded). A line y = a + b M_z over
		// the head's depth motion gives f = 1 / b and t_z = a / b. Biases common to both eyes cancel in s.
		auto interp = [&](double tq, WS &o) {
			size_t lo2 = 0, hi2 = ws.size();
			while (hi2 - lo2 > 1) { const size_t m = (lo2 + hi2) / 2; (ws[m].t <= tq ? lo2 : hi2) = m; }
			if (lo2 + 1 >= ws.size() || ws[lo2].t > tq) return false;
			const double span = ws[lo2 + 1].t - ws[lo2].t;
			if (span <= 0 || span > 0.08) return false;
			const float a = (float)((tq - ws[lo2].t) / span);
			o.t = tq;
			for (int i = 0; i < np; i++) o.p[i] = ws[lo2].p[i] * (1.0f - a) + ws[lo2 + 1].p[i] * a;
			return true;
		};
		auto eye_fit = [&](double f, const cv::Mat &rv, double &fo, double &tz, double &se_f, size_t &used, double &r2) {
			cv::Mat Rm;
			cv::Rodrigues(rv, Rm);
			const cv::Matx33d R(Rm);
			std::vector<double> xs, ys;
			for (const KS &k : ks) {
				WS w;
				if (std::isnan(k.X[0].z) || std::isnan(k.X[1].z) || !interp(k.t + best_dt, w)) continue;
				const cv::Vec3d E = R * cv::Vec3d(k.X[1].x - k.X[0].x, k.X[1].y - k.X[0].y, k.X[1].z - k.X[0].z);
				const cv::Vec3d M = R * cv::Vec3d(0.5 * (k.X[0].x + k.X[1].x), 0.5 * (k.X[0].y + k.X[1].y), 0.5 * (k.X[0].z + k.X[1].z));
				const cv::Point2f pm = (w.p[0] + w.p[1]) * 0.5f;
				const double xn = (pm.x - 319.5) / f, yn = (pm.y - 239.5) / f, s = cv::norm(w.p[1] - w.p[0]);
				if (s < 5) continue;
				xs.push_back(M[2]);
				ys.push_back(std::hypot(E[0] - xn * E[2], E[1] - yn * E[2]) / s);
			}
			std::vector<char> keep(xs.size(), 1);
			double a = 0, b = 0, sxx = 0, sse = 0;
			size_t n = 0;
			for (int it = 0; it < 4; it++) { /* least squares, then trim beyond 3 MAD, refit */
				double sx = 0, sy = 0, sxy = 0, sx2 = 0;
				n = 0;
				for (size_t i = 0; i < xs.size(); i++)
					if (keep[i]) { sx += xs[i]; sy += ys[i]; sxy += xs[i] * ys[i]; sx2 += xs[i] * xs[i]; n++; }
				if (n < 30) return false;
				const double mx = sx / (double)n;
				sxx = sx2 - sx * mx;
				b = (sxy - mx * sy) / sxx;
				a = sy / (double)n - b * mx;
				std::vector<double> res;
				for (size_t i = 0; i < xs.size(); i++) res.push_back(std::fabs(ys[i] - a - b * xs[i]));
				const double cut = 3.0 * 1.4826 * median(res);
				sse = 0;
				double syy = 0, my = sy / (double)n;
				for (size_t i = 0; i < xs.size(); i++) {
					keep[i] = res[i] <= cut;
					if (keep[i]) { sse += res[i] * res[i]; syy += (ys[i] - my) * (ys[i] - my); }
				}
				r2 = syy > 0 ? 1.0 - sse / syy : 0;
			}
			if (b <= 0) return false;
			const double se_b = std::sqrt(sse / (double)(n - 2) / sxx);
			fo = 1.0 / b;
			tz = a / b;
			se_f = se_b / (b * b);
			used = n;
			return true;
		};
		double f_eye = 0, se_eye = 0, tz_eye = 0;
		if (face) {
			double fcur = 690.0; /* start: Logitech spec 60 deg diagonal over the 4:3 sensor = 692.8 px */
			for (int it = 0; it < 3; it++) {
				cv::Mat rv, tv;
				std::vector<int> r;
				double med = 0, fo = 0, tz = 0, se = 0, r2 = 0;
				size_t used = 0;
				if (!pose_at(fcur, rv, tv, r, med) || !eye_fit(fcur, rv, fo, tz, se, used, r2)) { f_eye = 0; break; }
				std::printf("eye-scale fit %d (pose at f %.1f): f %.1f +- %.1f px, t_z %.1f mm, %zu samples, R^2 %.3f; PnP t_z %.1f mm\n", it, fcur,
					    fo, se, tz, used, r2, tv.at<double>(2));
				f_eye = fo;
				se_eye = se;
				tz_eye = tz;
				fcur = fo;
			}
		}
		// dense photometric agreement over the snapshots: every Kinect point (stride 2) projected into the
		// webcam; NCC of intensity and of gradient magnitude (edges must coincide), over all depths in view
		struct Smp { std::vector<cv::Point3f> P; std::vector<float> ki, kg; cv::Mat wi, wg; };
		std::vector<Smp> sm;
		for (const Snap &s : snaps) {
			Smp m;
			cv::Mat kg8, wg8, kgf, gx, gy, kmag;
			cv::cvtColor(s.k, kg8, cv::COLOR_BGR2GRAY);
			cv::cvtColor(s.w, wg8, cv::COLOR_BGR2GRAY);
			kg8.convertTo(kgf, CV_32F);
			wg8.convertTo(m.wi, CV_32F);
			cv::Sobel(kgf, gx, CV_32F, 1, 0);
			cv::Sobel(kgf, gy, CV_32F, 0, 1);
			cv::magnitude(gx, gy, kmag);
			cv::Sobel(m.wi, gx, CV_32F, 1, 0);
			cv::Sobel(m.wi, gy, CV_32F, 0, 1);
			cv::magnitude(gx, gy, m.wg);
			for (int y = 2; y < H - 2; y += 2)
				for (int x = 2; x < W - 2; x += 2) {
					const float *p = &s.pm[4 * ((size_t)y * W + x)];
					if (p[3] < 0.5f) continue;
					m.P.emplace_back(p[0], p[1], p[2]);
					m.ki.push_back(kgf.at<float>(y, x));
					m.kg.push_back(kmag.at<float>(y, x));
				}
			sm.push_back(std::move(m));
		}
		auto photo = [&](double f, const cv::Mat &rv, const cv::Mat &tv, double &ni, double &ng, size_t &cnt) {
			cv::Mat Kf(cv::Matx33d(f, 0, 319.5, 0, f, 239.5, 0, 0, 1));
			double s[2][5] = {{0}}; /* x, y, xx, yy, xy for intensity and gradient */
			cnt = 0;
			for (const Smp &m : sm) {
				std::vector<cv::Point2f> pr;
				cv::projectPoints(m.P, rv, tv, Kf, cv::noArray(), pr);
				for (size_t i = 0; i < pr.size(); i++) {
					const int u = (int)std::lround(pr[i].x), v = (int)std::lround(pr[i].y);
					if (u < 1 || v < 1 || u >= W - 1 || v >= H - 1) continue;
					const double a[2] = {m.ki[i], m.kg[i]}, b[2] = {m.wi.at<float>(v, u), m.wg.at<float>(v, u)};
					for (int c = 0; c < 2; c++) {
						s[c][0] += a[c]; s[c][1] += b[c]; s[c][2] += a[c] * a[c]; s[c][3] += b[c] * b[c]; s[c][4] += a[c] * b[c];
					}
					cnt++;
				}
			}
			double r[2] = {0, 0};
			for (int c = 0; c < 2 && cnt > 100; c++) {
				const double n = (double)cnt, cov = s[c][4] - s[c][0] * s[c][1] / n;
				const double va = s[c][2] - s[c][0] * s[c][0] / n, vb = s[c][3] - s[c][1] * s[c][1] / n;
				r[c] = cov / std::sqrt(va * vb);
			}
			ni = r[0];
			ng = r[1];
		};
		std::printf("focal profile (pose refined at each f): landmark pairs within %.0f px, their median error, "
			    "dense NCC intensity / gradient over %zu snapshots:\n", sel, sm.size());
		double fbest = 0, mbest = 0, gbest = -2, flm = 0;
		size_t nbest = 0, nlm = 0;
		std::vector<std::pair<int, size_t>> prof;
		std::vector<std::pair<int, double>> gprof;
		for (int f = 400; f <= 1300; f += 25) {
			cv::Mat rv, tv;
			std::vector<int> r;
			double med = 0, ni = 0, ng = 0;
			size_t pc = 0;
			if (!pose_at(f, rv, tv, r, med)) continue;
			photo(f, rv, tv, ni, ng, pc);
			prof.emplace_back(f, r.size());
			gprof.emplace_back(f, ng);
			std::printf("  f %4d: %5zu  %.3f px   NCC %.4f / %.4f on %zu px\n", f, r.size(), med, ni, ng, pc);
			if (r.size() > nlm) { nlm = r.size(); flm = f; }
			if (ng > gbest) { gbest = ng; fbest = f; nbest = r.size(); mbest = med; }
		}
		if (!nlm) { std::printf("FAIL no pose at any focal length\n"); return 1; }
		double glo = gbest, ghi = gbest;
		for (auto &p : gprof) {
			if (p.first == (int)fbest - 100) glo = p.second;
			if (p.first == (int)fbest + 100) ghi = p.second;
		}
		std::printf("dense profile: best f %.0f px, gradient NCC %.4f; 100 px lower %.4f, 100 px higher %.4f; landmark-count best was f %.0f\n",
			    fbest, gbest, glo, ghi, flm);
		size_t lo = nbest, hi = nbest; /* sharpness: counts 100 px either side of the best */
		for (auto &p : prof) {
			if (p.first == (int)fbest - 100) lo = p.second;
			if (p.first == (int)fbest + 100) hi = p.second;
		}
		std::printf("focal profile: best f %.0f px (%zu pairs, median %.3f px); 100 px lower keeps %.1f %%, 100 px higher %.1f %%\n", fbest,
			    nbest, mbest, 100.0 * (double)lo / (double)nbest, 100.0 * (double)hi / (double)nbest);
		if (f_fixed > 0) { /* independent focal length (spec / earlier measurement) */
			fbest = f_fixed;
			std::printf("using the fixed focal length %.2f px%s\n", f_fixed, face && f_eye > 0 ? " (eye-scale fit above is the cross-check)" : "");
		} else if (face && f_eye > 0) { /* measured focal length wins over both flat profiles */
			fbest = std::round(f_eye * 10.0) / 10.0;
			std::printf("using the eye-scale focal length %.1f +- %.1f px (t_z %.1f mm)\n", f_eye, se_eye, tz_eye);
		}
		double med = 0;
		pose_at(fbest, rvs[0], tvs[0], inl, med);
		K = cv::Mat(cv::Matx33d(fbest, 0, 319.5, 0, fbest, 239.5, 0, 0, 1));
		std::vector<cv::Point2f> pr;
		cv::projectPoints(obj, rvs[0], tvs[0], K, D, pr);
		double ss = 0;
		for (int i : inl) ss += std::pow(cv::norm(pr[(size_t)i] - img[(size_t)i]), 2);
		rms = std::sqrt(ss / (double)inl.size());
		std::printf("pose at f %.0f: %zu inliers within %.0f px, rms %.3f px\n", fbest, inl.size(), sel, rms);
		{ /* bootstrap: 20 resolves at this focal length on random 70 % subsets of the pairs */
			cv::RNG rng(12345);
			const cv::Mat Kb(cv::Matx33d(fbest, 0, 319.5, 0, fbest, 239.5, 0, 0, 1));
			std::vector<cv::Vec3d> cs;
			std::vector<double> angs;
			for (int b = 0; b < 20; b++) {
				std::vector<cv::Point3f> O;
				std::vector<cv::Point2f> I;
				for (size_t i = 0; i < obj.size(); i++)
					if (rng.uniform(0.0, 1.0) < 0.7) { O.push_back(obj[i]); I.push_back(img[i]); }
				cv::Mat rv, tv, Rb;
				std::vector<int> r;
				if (!cv::solvePnPRansac(O, I, Kb, cv::noArray(), rv, tv, false, 1000, face ? 6.0f : 4.0f, 0.999, r) || r.size() < 12) continue;
				std::vector<cv::Point3f> O2;
				std::vector<cv::Point2f> I2;
				for (int i : r) { O2.push_back(O[(size_t)i]); I2.push_back(I[(size_t)i]); }
				cv::solvePnPRefineLM(O2, I2, Kb, cv::noArray(), rv, tv);
				cv::Rodrigues(rv, Rb);
				cv::Mat Cb = -Rb.t() * tv;
				cs.emplace_back(Cb.at<double>(0), Cb.at<double>(1), Cb.at<double>(2));
				angs.push_back(cv::norm(rv) * 180.0 / M_PI);
			}
			cv::Vec3d m(0, 0, 0), sd(0, 0, 0);
			double ma = 0, sa = 0;
			for (size_t i = 0; i < cs.size(); i++) { m += cs[i]; ma += angs[i]; }
			m /= (double)cs.size();
			ma /= (double)cs.size();
			for (size_t i = 0; i < cs.size(); i++) {
				for (int k = 0; k < 3; k++) sd[k] += (cs[i][k] - m[k]) * (cs[i][k] - m[k]);
				sa += (angs[i] - ma) * (angs[i] - ma);
			}
			for (int k = 0; k < 3; k++) sd[k] = std::sqrt(sd[k] / (double)(cs.size() - 1));
			std::printf("bootstrap (%zu of 20 resolves at f %.2f): centre %.1f %.1f %.1f mm, sd %.1f %.1f %.1f mm; rotation %.2f deg, sd %.2f deg\n",
				    cs.size(), fbest, m[0], m[1], m[2], sd[0], sd[1], sd[2], ma, std::sqrt(sa / (double)(cs.size() - 1)));
		}
	} else {
		// SIFT pairs on a static scene: calibrateCamera (fx = fy, principal point, k1, k2) on the inliers
		const int flags = cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K3;
		try {
			for (int round = 0; round < 3; round++) {
				std::vector<std::vector<cv::Point3f>> O(1);
				std::vector<std::vector<cv::Point2f>> I(1);
				for (int i : inl) { O[0].push_back(obj[(size_t)i]); I[0].push_back(img[(size_t)i]); }
				rms = cv::calibrateCamera(O, I, cv::Size(W, H), K, D, rvs, tvs, flags);
				std::vector<cv::Point2f> pr;
				cv::projectPoints(obj, rvs[0], tvs[0], K, D, pr);
				inl.clear();
				for (size_t i = 0; i < obj.size(); i++)
					if (cv::norm(pr[i] - img[i]) < sel) inl.push_back((int)i);
				std::printf("refine %d: rms %.3f px on %zu points; %zu within %.0f px under the refined model\n", round, rms,
					    O[0].size(), inl.size(), sel);
				if (inl.size() < 30) break;
			}
		} catch (const cv::Exception &e) {
			std::printf("FAIL calibrateCamera: %s\n", e.what());
			return 1;
		}
	}

	cv::Mat R;
	cv::Rodrigues(rvs[0], R);
	cv::Mat t = tvs[0], C = -R.t() * t;
	const double ang = cv::norm(rvs[0]) * 180.0 / M_PI, dist = cv::norm(C);
	const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
	std::printf("result: f %.2f %.2f px, centre %.2f %.2f, k1 %.5f k2 %.5f\n", fx, fy, cx, cy, D.at<double>(0), D.at<double>(1));
	std::printf("result: webcam centre in the Kinect frame (X right, Y down, Z forward) %.1f %.1f %.1f mm, |C| %.1f mm, rotation %.2f deg\n",
		    C.at<double>(0), C.at<double>(1), C.at<double>(2), dist, ang);
	struct { const char *what; bool ok; } chk[] = {
		{"inliers >= 50", inl.size() >= 50},
		{face ? "rms <= 3.5 px" : track ? "rms <= 2.5 px" : "rms <= 1.5 px", rms <= (face ? 3.5 : track ? 2.5 : 1.5)},
		{"webcam above the Kinect (C.y < 0)", C.at<double>(1) < 0},
		{track ? "20 mm <= |C| <= 1500 mm" : "20 mm <= |C| <= 300 mm", dist >= 20 && dist <= (track ? 1500 : 300)},
		{track ? "rotation < 75 deg" : "rotation < 20 deg", ang < (track ? 75 : 20)},
		{"350 <= f <= 1300 px", fx >= 350 && fx <= 1300},
		{"principal point within 60 px of the centre", std::fabs(cx - 319.5) < 60 && std::fabs(cy - 239.5) < 60},
	};
	int fails = 0;
	for (auto &c : chk) { std::printf("%s %s\n", c.ok ? "PASS" : "FAIL", c.what); fails += !c.ok; }

	// colour transfer: kinect = gain * webcam + offset, per channel (0..1), trimmed least squares
	std::vector<cv::Point3f> P;
	std::vector<cv::Vec3f> kc;
	std::vector<size_t> owner;
	for (size_t si = 0; si < snaps.size(); si++)
		for (int y = 0; y < H; y += 2)
			for (int x = 0; x < W; x += 2) {
				const float *p = &snaps[si].pm[4 * ((size_t)y * W + x)];
				if (p[3] < 0.5f) continue;
				P.emplace_back(p[0], p[1], p[2]);
				const cv::Vec3b b = snaps[si].k.at<cv::Vec3b>(y, x);
				kc.emplace_back(b[2] / 255.f, b[1] / 255.f, b[0] / 255.f);
				owner.push_back(si);
			}
	std::vector<cv::Point2f> pr;
	cv::projectPoints(P, rvs[0], tvs[0], K, D, pr);
	double gain[3] = {1, 1, 1}, off[3] = {0, 0, 0};
	for (int ch = 0; ch < 3; ch++) {
		std::vector<std::pair<double, double>> xy; // (webcam, kinect)
		for (size_t i = 0; i < P.size(); i++) {
			const int u = (int)std::lround(pr[i].x), v = (int)std::lround(pr[i].y);
			if (u < 1 || v < 1 || u >= W - 1 || v >= H - 1) continue;
			const double wv = snaps[owner[i]].w.at<cv::Vec3b>(v, u)[2 - ch] / 255.0, kv = kc[i][ch];
			if (wv > 0.03 && wv < 0.97 && kv > 0.03 && kv < 0.97) xy.emplace_back(wv, kv);
		}
		for (int it = 0; it < 3 && xy.size() > 100; it++) {
			double sx = 0, sy = 0, sxx = 0, sxy = 0;
			for (auto &q : xy) { sx += q.first; sy += q.second; sxx += q.first * q.first; sxy += q.first * q.second; }
			const double n = (double)xy.size(), den = n * sxx - sx * sx;
			if (std::fabs(den) < 1e-9) break;
			gain[ch] = (n * sxy - sx * sy) / den;
			off[ch] = (sy - gain[ch] * sx) / n;
			std::vector<double> r;
			for (auto &q : xy) r.push_back(std::fabs(q.second - gain[ch] * q.first - off[ch]));
			const double mad = median(r), cut = 2.5 * 1.4826 * mad;
			std::vector<std::pair<double, double>> keep;
			for (size_t i = 0; i < xy.size(); i++)
				if (r[i] <= cut) keep.push_back(xy[i]);
			xy.swap(keep);
		}
		std::printf("colour %c: kinect = %.4f * webcam %+.4f (%zu samples after trimming)\n", "RGB"[ch], gain[ch], off[ch], xy.size());
	}

	if (fails) { std::printf("NOT WRITTEN: %d physical check(s) failed\n", fails); return 2; }
	FILE *o = std::fopen(argv[2], "w");
	if (!o) { std::perror(argv[2]); return 1; }
	const std::time_t now = std::time(nullptr);
	char stamp[64];
	std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	std::fprintf(o, "# kinect3d webcam calibration, %s, from %s (%zu snapshots)\n", stamp, dir.c_str(), snaps.size());
	std::fprintf(o, "# %s: %zu correspondences, %zu inliers, rms %.3f px, focal scan best %d px, time offset %+.1f ms\n", track ? mode : "SIFT", obj.size(), inl.size(), rms, best_f, best_dt * 1e3);
	std::fprintf(o, "# webcam centre in the Kinect frame %.1f %.1f %.1f mm, rotation %.2f deg\n", C.at<double>(0), C.at<double>(1), C.at<double>(2), ang);
	std::fprintf(o, "K %.6f %.6f %.6f %.6f\nD %.8f %.8f\nR", fx, fy, cx, cy, D.at<double>(0), D.at<double>(1));
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++) std::fprintf(o, " %.9f", R.at<double>(r, c));
	std::fprintf(o, "\nt %.4f %.4f %.4f\n", t.at<double>(0), t.at<double>(1), t.at<double>(2));
	std::fprintf(o, "gain %.5f %.5f %.5f\noffset %.5f %.5f %.5f\n", gain[0], gain[1], gain[2], off[0], off[1], off[2]);
	std::fclose(o);
	std::printf("written %s\n", argv[2]);
	return 0;
}
