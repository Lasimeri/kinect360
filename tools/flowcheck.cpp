// flowcheck.cpp: oracle and plain-form baseline for the GPU flow (src/flow.c).
// Reads DIR/NAME-previous.pgm, DIR/NAME-newest.pgm (the exact 8-bit frames the GPU saw) and
// DIR/NAME-flow.f32 (GPU result: per pixel u, v, fb error, valid; backward newest -> previous).
//  1. OpenCV calcOpticalFlowPyrLK, same parameters (11x11, 3 extra levels, 10 iterations, 0.01 px),
//     newest -> previous at every 8th pixel >= 16 px from the border: endpoint difference vs GPU.
//  2. CPU timing of the plain forms on the same pair: dense PyrLK (all 307,200 pixels), DIS MEDIUM, DIS FAST.
//  3. GPU vs DIS MEDIUM endpoint difference on GPU-valid pixels (different algorithm: informational).
// Build: g++ -O2 -std=c++17 -Wall -Wextra -Werror -I/usr/include/opencv5 tools/flowcheck.cpp -o out/flowcheck
//        -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static double pct(std::vector<double> v, double p)
{
	if (v.empty()) return NAN;
	size_t i = std::min(v.size() - 1, (size_t)(p * (double)v.size()));
	std::nth_element(v.begin(), v.begin() + (long)i, v.end());
	return v[i];
}

template <class F> static double median_ms(int runs, F fn)
{
	std::vector<double> t;
	for (int r = 0; r < runs; r++) {
		auto a = std::chrono::steady_clock::now();
		fn();
		auto b = std::chrono::steady_clock::now();
		t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
	}
	return pct(t, 0.5);
}

int main(int argc, char **argv)
{
	if (argc < 3) { std::fprintf(stderr, "usage: %s DIR NAME\n", argv[0]); return 64; }
	const std::string base = std::string(argv[1]) + "/" + argv[2];
	cv::Mat prev = cv::imread(base + "-previous.pgm", cv::IMREAD_GRAYSCALE), cur = cv::imread(base + "-newest.pgm", cv::IMREAD_GRAYSCALE);
	if (prev.empty() || cur.empty() || prev.size() != cur.size()) { std::fprintf(stderr, "cannot read the frame pair\n"); return 1; }
	const int W = cur.cols, H = cur.rows;
	std::vector<float> g((size_t)W * H * 4);
	std::ifstream in(base + "-flow.f32", std::ios::binary);
	if (!in.read(reinterpret_cast<char *>(g.data()), (std::streamsize)(g.size() * sizeof(float)))) { std::fprintf(stderr, "cannot read the GPU flow\n"); return 1; }
	double mdiff = cv::norm(cur, prev, cv::NORM_L1) / (W * H);
	std::printf("%s: %dx%d, mean |newest - previous| %.3f gray levels, OpenCV threads %d\n", argv[2], W, H, mdiff, cv::getNumThreads());

	// 1. oracle
	std::vector<cv::Point2f> pts, out;
	for (int y = 16; y < H - 16; y += 8)
		for (int x = 16; x < W - 16; x += 8) pts.emplace_back((float)x, (float)y);
	std::vector<uchar> st;
	std::vector<float> err;
	const cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.01);
	cv::calcOpticalFlowPyrLK(cur, prev, pts, out, st, err, cv::Size(11, 11), 3, tc);
	std::vector<double> d, mag;
	unsigned cvok = 0, gpuok = 0;
	for (size_t i = 0; i < pts.size(); i++) {
		const int x = (int)pts[i].x, y = (int)pts[i].y;
		const float *f = &g[4 * ((size_t)y * W + x)];
		const bool gv = f[3] > 0.5f;
		cvok += st[i];
		gpuok += gv;
		if (st[i] && gv) {
			const double u = out[i].x - pts[i].x, v = out[i].y - pts[i].y;
			d.push_back(std::hypot(u - f[0], v - f[1]));
			mag.push_back(std::hypot(u, v));
		}
	}
	size_t within = (size_t)std::count_if(d.begin(), d.end(), [](double e) { return e <= 0.1; });
	std::printf("oracle PyrLK: %zu grid points, OpenCV ok %u, GPU valid %u, both %zu\n", pts.size(), cvok, gpuok, d.size());
	std::printf("oracle PyrLK: |GPU - OpenCV| median %.4f  p90 %.4f  p99 %.4f  max %.4f px; <= 0.1 px: %.1f %%; |flow| median %.3f p90 %.3f px\n",
		    pct(d, 0.5), pct(d, 0.9), pct(d, 0.99), d.empty() ? NAN : *std::max_element(d.begin(), d.end()),
		    d.empty() ? 0.0 : 100.0 * (double)within / (double)d.size(), pct(mag, 0.5), pct(mag, 0.9));

	// 2. plain-form CPU timing on the same pair
	std::vector<cv::Point2f> all, allout;
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) all.emplace_back((float)x, (float)y);
	double t_lk = median_ms(5, [&] { cv::calcOpticalFlowPyrLK(cur, prev, all, allout, st, err, cv::Size(11, 11), 3, tc); });
	cv::Mat dis_m, dis_f;
	auto dm = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
	auto df = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
	double t_dm = median_ms(10, [&] { dm->calc(cur, prev, dis_m); });
	double t_df = median_ms(10, [&] { df->calc(cur, prev, dis_f); });
	std::printf("CPU plain forms, one direction, median: dense PyrLK %.2f ms, DIS MEDIUM %.2f ms, DIS FAST %.2f ms\n", t_lk, t_dm, t_df);

	// 3. GPU vs DIS (different algorithm)
	std::vector<double> e;
	for (int y = 16; y < H - 16; y++)
		for (int x = 16; x < W - 16; x++) {
			const float *f = &g[4 * ((size_t)y * W + x)];
			if (f[3] < 0.5f) continue;
			const cv::Point2f p = dis_m.at<cv::Point2f>(y, x);
			e.push_back(std::hypot(p.x - f[0], p.y - f[1]));
		}
	std::printf("GPU vs DIS MEDIUM on %zu GPU-valid pixels: median %.3f  p90 %.3f px\n", e.size(), pct(e, 0.5), pct(e, 0.9));
	return 0;
}
