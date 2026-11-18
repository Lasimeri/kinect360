// facetrack.cpp: see facetrack.h. YuNet row = box x y w h, 5 landmarks (x, y), score.
#include "facetrack.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <cstdio>

struct facetrack {
	cv::Ptr<cv::FaceDetectorYN> det;
	int w, h;
	cv::Mat bgr;
};

extern "C" facetrack *ft_create(const char *model, int w, int h, int threads)
{
	try {
		if (threads > 0) cv::setNumThreads(threads);
		facetrack *f = new facetrack;
		f->det = cv::FaceDetectorYN::create(model, "", cv::Size(w, h), 0.7f, 0.3f, 5);
		f->w = w;
		f->h = h;
		return f;
	} catch (const cv::Exception &e) {
		std::fprintf(stderr, "facetrack: %s\n", e.what());
		return nullptr;
	}
}

extern "C" int ft_detect(facetrack *f, const uint8_t *rgb, face5 *best)
{
	try {
		cv::Mat in(f->h, f->w, CV_8UC3, const_cast<uint8_t *>(rgb)), faces;
		cv::cvtColor(in, f->bgr, cv::COLOR_RGB2BGR); // YuNet is trained on BGR
		f->det->detect(f->bgr, faces);
		if (faces.empty()) return 0;
		int b = 0;
		for (int i = 1; i < faces.rows; i++)
			if (faces.at<float>(i, 14) > faces.at<float>(b, 14)) b = i;
		const float *r = faces.ptr<float>(b);
		best->score = r[14];
		for (int k = 0; k < 4; k++) best->box[k] = r[k];
		for (int k = 0; k < 10; k++) best->lm[k] = r[4 + k];
		return faces.rows;
	} catch (const cv::Exception &e) {
		std::fprintf(stderr, "facetrack: %s\n", e.what());
		return -1;
	}
}

extern "C" void ft_destroy(facetrack *f) { delete f; }
