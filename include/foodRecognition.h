#ifndef FOOD_RECOGNITION_H
#define FOOD_RECOGNITION_H
#endif

#include <opencv2/opencv.hpp>

// class for food recognition
double featMatching(cv::Mat img1, cv::Mat img2, cv::Mat &Result_MATCHING);
double histMatching(cv::Mat img1, cv::Mat img2, cv::Mat &Result_MATCHING);
