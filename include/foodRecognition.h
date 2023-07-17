#ifndef FOOD_RECOGNITION_H
#define FOOD_RECOGNITION_H
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include<vector>


double calculateIoU(const cv::Rect& bbox1, const cv::Rect& groundTruth);


//double calculateAP(const std::vector<cv::Rect>& gtBboxes, const std::vector<cv::Rect>& predBboxes, double iouThreshold);

double meanAveragePrecision();


double foodLeftoverEstimation(const cv::Mat& reference, const cv::Mat& image, const cv::Rect& bbox1, const cv::Rect& bbox2);