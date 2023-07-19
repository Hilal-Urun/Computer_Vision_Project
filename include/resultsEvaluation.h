#ifndef RESULTS_EVALUATION_H
#define RESULTS_EVALUATION_H
#endif

#include <opencv2/opencv.hpp>

// class for results evaluation
double calculateIoU(const cv::Rect& bbox1, const cv::Rect& groundTruth);
double calculateMeanIoU(const std::vector<cv::Rect>& gtBboxes, const std::vector<cv::Rect>& predBboxes);
double foodLeftoverEstimation(const cv::Mat& before, const cv::Mat& after, const cv::Rect& bboxBefore, const cv::Rect& bboxAfter);
