#ifndef FOOD_RECOGNITION_H
#define FOOD_RECOGNITION_H
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include<vector>


// class for food recognition
class FoodRecognition
{
public:
	FoodRecognition(cv::Mat img);
	~FoodRecognition();
	cv::Mat getFoodImage();
	cv::Mat runCanny(int lowThreshold, int highThreshold);
	void drawHistogram(cv::Mat img);

private:

	cv::Mat foodImage;
	cv::Mat cannyResult;
};