#ifndef FOOD_SEGMENTATION_H
#define FOOD_SEGMENTATION_H
#endif


#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "foodRecognition.h"
#include "leftoverEst.h"
#include <opencv2/core/hal/interface.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<opencv2/opencv_modules.hpp>


cv::Rect getBoundingRectangle(cv::Rect rect1, cv::Rect rect2);


bool rectanglesIntersect(cv::Rect rect1, cv::Rect rect2);


cv::Mat GrabcutAlgorithm(const cv::Mat& src, const cv::Rect& boundingBox);