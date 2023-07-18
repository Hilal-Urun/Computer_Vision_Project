#ifndef FOOD_RECOGNITION_H
#define FOOD_RECOGNITION_H
#endif

#include <opencv2/opencv.hpp>

// functions for food recognition
bool rectanglesIntersect(cv::Rect rect1, cv::Rect rect2);
cv::Rect getBoundingRectangle(cv::Rect rect1, cv::Rect rect2);
cv::Mat GrabcutAlgorithm(const cv::Mat& src, const cv::Rect& boundingBox);
double featMatching(cv::Mat img1, cv::Mat img2, cv::Mat &Result_MATCHING);
double histMatching(cv::Mat img1, cv::Mat img2);
std::vector<std::tuple<int, int>> boxesMatch(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, std::vector<cv::Rect> boundingBoxes1, std::vector<cv::Rect> boundingBoxes2);
double foodLeftoverEstimation(const cv::Mat& before, const cv::Mat& after, const cv::Rect& bboxBefore, const cv::Rect& bboxAfter);
