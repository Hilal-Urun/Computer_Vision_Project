#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "foodRecognition.h"
#include "foodSegmentation.h"
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


//bilateral filter because its smoothing when preserve the edges
cv::Mat applyBilateralFilter(const cv::Mat& inputImage) {
	cv::Mat filteredImage;
	cv::bilateralFilter(inputImage, filteredImage, 5, 75, 75);
	return filteredImage;
}

cv::Mat applyContrastEnhancement(const cv::Mat& filteredImage) {
	cv::Mat enhancedImage;
	filteredImage.copyTo(enhancedImage);
	// Convert the filtered image to the YCrCb color space
	//cvtColor(filteredImage, enhancedImage, cv::COLOR_BGR2YCrCb);
	// Split the YCrCb image into individual channels
	cv::Mat channels[3];
	split(enhancedImage, channels);

	// Apply histogram equalization to the Y channel

	equalizeHist(channels[0], channels[0]);
	equalizeHist(channels[1], channels[1]);
	equalizeHist(channels[2], channels[2]);

	std::vector<cv::Mat> ccc = { channels[0],channels[1],channels[2] };
	// Merge the enhanced channels back into the YCrCb image
	merge(ccc, enhancedImage);

	// Convert the enhanced image back to the BGR color space
	//cvtColor(enhancedImage, enhancedImage, cv::COLOR_YCrCb2BGR);

	return enhancedImage;
}


//Hist eq also do same thing, we can ignore it ?

cv::Mat applyColorAdjustment(const cv::Mat& inputImage) {
	cv::Mat adjustedImage;
	// Convert the image to the LAB color space
	cvtColor(inputImage, adjustedImage, cv::COLOR_BGR2Lab);

	// Split the LAB image into L, a, and b channels
	std::vector<cv::Mat> labChannels(3);
	split(adjustedImage, labChannels);

	// Apply color adjustment on the a and b channels
	labChannels[1] += 10; // Example: increase the a channel by 10
	labChannels[2] -= 10; // Example: decrease the b channel by 10

	// Merge the modified channels back to LAB image
	merge(labChannels, adjustedImage);

	// Convert the LAB image back to the BGR color space
	cvtColor(adjustedImage, adjustedImage, cv::COLOR_Lab2BGR);
	return adjustedImage;
}


std::vector<cv::Vec3f> houghTransform(cv::Mat& img) {
	std::vector<cv::Vec3f> circles;
	cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0);
	medianBlur(img, img, 3);
	cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1,
		200,  // change this value to detect circles with different distances to each other
		200, 100, 0, 0 // change the last two parameters
   // (min_radius & max_radius) to detect larger circles
	);

	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Vec3i c = circles[i];
		cv::Point center = cv::Point(c[0], c[1]);
		// circle center
		cv::circle(img, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
		// circle outline
		int radius = c[2];
		circle(img, center, radius, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
		cv::imshow("Hough", img);
	}
	return circles;
}

std::vector<cv::KeyPoint> surfDetection(cv::Mat& img) {

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img, keypoints);
	//-- Draw keypoints
	cv::Mat img_keypoints;
	drawKeypoints(img, keypoints, img_keypoints);

	//-- Show detected (drawn) keypoints
	imshow("SURF Keypoints", img_keypoints);
	return keypoints;
}

void siftDetection(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
	//std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(img, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(img, keypoints, output);
	cv::imshow("sift_result", output);
	std::cout << keypoints.size() << std::endl;
	//return keypoints;
	siftPtr->clear();
}


void featureMatching(cv::Mat& img_1, cv::Mat& img_2) {

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SiftDescriptorExtractor extractor;
	cv::Mat descriptors_1, descriptors_2;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	
	siftDetection(img_1, keypoints_1);

	siftDetection(img_2, keypoints_2);
	//std::vector<cv::KeyPoint> keypoints_2 = siftDetection(img_2);
	std::cout << keypoints_1.size();
	cv::cvtColor(img_1, img_1, cv::COLOR_RGB2GRAY);
	cv::cvtColor(img_2, img_2, cv::COLOR_RGB2GRAY);
	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< cv::DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= std::max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	cv::Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	imshow("Good Matches", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

}


int main(int argc, char** argv)
{
	// loading and input validation
	std::string trayNumber;
	while (true) {
		std::cout << "Tray number (1 to 8):" << std::endl;
		std::cin >> trayNumber;
		if (isdigit(trayNumber[0]) && stoi(trayNumber) >= 1 && stoi(trayNumber) <= 8){
			break;
		}
	}
	std::string imagePath = "..//dataset//tray" + trayNumber;
	cv::Mat foodImage, foodLeftover1, foodLeftover2, foodLeftover3;
	foodImage = cv::imread(imagePath + "//food_image.jpg", cv::IMREAD_COLOR);
	foodLeftover1 = cv::imread(imagePath + "//leftover1.jpg", cv::IMREAD_COLOR);
	foodLeftover2 = cv::imread(imagePath + "//leftover2.jpg", cv::IMREAD_COLOR);
	foodLeftover3 = cv::imread(imagePath + "//leftover3.jpg", cv::IMREAD_COLOR);
	// Check if the images are loaded correctly:
	if (foodImage.empty() || foodLeftover1.empty() ||
		foodLeftover2.empty() || foodLeftover3.empty()) {
		std::cerr << "Could not load the image: " << std::endl;
		return 1;
	}
	//cv::namedWindow("Original_image", cv::WINDOW_NORMAL);
	//cv::namedWindow("Leftover_1", cv::WINDOW_NORMAL);
	//cv::namedWindow("Leftover_2", cv::WINDOW_NORMAL);
	//cv::namedWindow("Leftover_3", cv::WINDOW_NORMAL);
	//imshow("Original_image", foodImage);
	//imshow("Leftover_1", foodLeftover1);
	//imshow("Leftover_2", foodLeftover2);
	//imshow("Leftover_3", foodLeftover3);

	FoodRecognition recognition(foodImage);
	cv::Mat cannyRes = recognition.runCanny(20, 60);

	cv::Mat filteredImage = applyBilateralFilter(foodImage);
	cv::Mat enhancedImage = applyContrastEnhancement(filteredImage);
	cv::Mat colorAdjustedImage = applyColorAdjustment(enhancedImage);
	//imshow("bilateral", filteredImage);
	//imshow("contrast", enhancedImage);
	//imshow("colorAdjustment", filteredImage);
	
	//surfDetection(filteredImage);
	//siftDetection(filteredImage);

	
	cv::Mat filteredImageCopy;

	filteredImage.copyTo(filteredImageCopy);
	std::vector<cv::Vec3f> circles = houghTransform(filteredImage);
	// try hough transform on canny


	// Setup a rectangle to define your region of interest
	cv::Rect myROI(circles[0][0] - circles[0][2], 0, circles[0][2] * 2, filteredImage.rows);

	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	cv::Mat croppedRef(filteredImageCopy, myROI);

	cv::Mat cropped;
	// Copy the data into new matrix
	croppedRef.copyTo(cropped);

	imshow("Cropped", cropped);
	//featureMatching(cropped, foodLeftover1);


	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
	std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(foodImage, keypoints);


	cv::Ptr<cv::SIFT> siftPtr2 = cv::SIFT::create();
	std::vector<cv::KeyPoint> keypoints2;
	siftPtr2->detect(cropped, keypoints2);


	cv::SiftDescriptorExtractor extractor;
	cv::Mat descriptors_1, descriptors_2;


	extractor.compute(foodImage, keypoints, descriptors_1);
	extractor.compute(cropped, keypoints2, descriptors_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< cv::DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= std::max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	cv::Mat img_matches;
	drawMatches(foodImage, keypoints, cropped, keypoints2,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	imshow("Good Matches", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}




	/*
	cv::Mat croppedFloat;
	cropped.convertTo(croppedFloat, CV_32FC3, 1. / 255);
	imshow("float", croppedFloat);
	//cv::imshow("cropped", cropped);
	cv::Mat labels;
	//cv::kmeans(cropped, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 10, cv::KMEANS_PP_CENTERS);
		//imshow("k-means", cropped);
	cv::Mat mask;
	cv::inRange(cropped, cv::Scalar(60, 60, 60), cv::Scalar(255, 255, 255), mask);
	cropped.setTo(cv::Scalar(0, 0, 0), mask);
	cv::imshow("cropped", cropped);


	// convert to binary
	cv::Mat bw;
	cvtColor(cropped, bw, cv::COLOR_BGR2GRAY);
	threshold(bw, bw, 60, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	imshow("binary", bw);


	// Perform the distance transform algorithm
	cv::Mat dist;
	distanceTransform(bw, dist, cv::DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	imshow("Distance Transform Image", dist);

	threshold(dist, dist, 0.3, 1.0, cv::THRESH_BINARY);
	// Dilate a bit the dist image
	cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	imshow("Peaks", dist);
	*/
	cv::waitKey(0);
	return 0;
}