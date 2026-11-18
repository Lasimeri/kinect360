// facetest.cpp: runs src/facetrack on snapshot pairs (DIR/snapN-kinect.ppm, -webcam.ppm, -points.f32):
// faces found per view, score, landmarks, 3D of the Kinect landmarks (point map, nearest valid within 3 px),
// detection time (median of 20), and DIR/faces-snapN.png with the landmarks drawn for an eye check.
#include "../src/facetrack.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
	if (argc < 3) { std::fprintf(stderr, "usage: %s MODEL DIR\n", argv[0]); return 64; }
	const int W = 640, H = 480;
	facetrack *ft = ft_create(argv[1], W, H, 4);
	if (!ft) return 1;
	const char *name[5] = {"right eye", "left eye", "nose tip", "right mouth", "left mouth"};
	for (int s = 0; s < 16; s++) {
		const std::string base = std::string(argv[2]) + "/snap" + std::to_string(s);
		cv::Mat k = cv::imread(base + "-kinect.ppm"), w = cv::imread(base + "-webcam.ppm");
		if (k.empty() || w.empty()) break;
		std::vector<float> pm((size_t)W * H * 4);
		std::ifstream in(base + "-points.f32", std::ios::binary);
		const bool have_pm = (bool)in.read(reinterpret_cast<char *>(pm.data()), (std::streamsize)(pm.size() * 4));
		cv::Mat out(H, 2 * W, CV_8UC3);
		const cv::Mat *img[2] = {&k, &w};
		for (int c = 0; c < 2; c++) {
			cv::Mat rgb;
			cv::cvtColor(*img[c], rgb, cv::COLOR_BGR2RGB);
			face5 f;
			std::vector<double> ms;
			int n = 0;
			for (int r = 0; r < 20; r++) {
				auto a = std::chrono::steady_clock::now();
				n = ft_detect(ft, rgb.data, &f);
				ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
			}
			std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
			img[c]->copyTo(out(cv::Rect(c * W, 0, W, H)));
			std::printf("snap %d %s: %d face(s), %.2f ms median", s, c ? "webcam" : "kinect", n, ms[10]);
			if (n <= 0) { std::printf("\n"); continue; }
			std::printf(", score %.3f, box %.0f %.0f %.0fx%.0f\n", f.score, f.box[0], f.box[1], f.box[2], f.box[3]);
			for (int l = 0; l < 5; l++) {
				const float x = f.lm[2 * l], y = f.lm[2 * l + 1];
				cv::circle(out, cv::Point2f(x + (float)(c * W), y), 4, cv::Scalar(0, 255 - 50 * l, 50 * l), -1);
				std::printf("   %-11s %7.2f %7.2f", name[l], x, y);
				if (c == 0 && have_pm) {
					bool ok = false;
					float best = 1e9f, X[3] = {0, 0, 0};
					for (int dy = -3; dy <= 3; dy++)
						for (int dx = -3; dx <= 3; dx++) {
							const int px = (int)(x + 0.5f) + dx, py = (int)(y + 0.5f) + dy;
							if (px < 0 || py < 0 || px >= W || py >= H) continue;
							const float *p = &pm[4 * ((size_t)py * W + px)];
							const float d2 = (float)(dx * dx + dy * dy);
							if (p[3] > 0.5f && d2 < best) { best = d2; X[0] = p[0]; X[1] = p[1]; X[2] = p[2]; ok = true; }
						}
					if (ok) std::printf("   3D %8.1f %8.1f %8.1f mm", X[0], X[1], X[2]);
					else std::printf("   3D none within 3 px");
				}
				std::printf("\n");
			}
		}
		cv::Mat small;
		cv::resize(out, small, cv::Size(W, H / 2));
		cv::imwrite(std::string(argv[2]) + "/faces-snap" + std::to_string(s) + ".png", small);
	}
	ft_destroy(ft);
	return 0;
}
