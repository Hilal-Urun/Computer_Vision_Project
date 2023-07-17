#include <opencv2/opencv.hpp>
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
#include <iostream>
#include <string>
#include <fstream>
#include "foodRecognition.h"
#include "foodSegmentation.h"
#include "leftoverEst.h"

std::vector<cv::Rect> readBoudingBox(const std::string& imagePath) {

	std::ifstream inputFile(imagePath);

	if (!inputFile)
	{
		std::cout << "Failed to open the file." << std::endl;
		std::vector<cv::Rect> emptyVector;
		return emptyVector;
	}

	std::vector<int> numbers;

	std::string line;
	while (std::getline(inputFile, line))
	{
		std::string token;
		size_t pos = 0;

		while ((pos = line.find(',')) != std::string::npos)
		{
			token = line.substr(0, pos);
			numbers.push_back(std::stoi(token)); // Convert token to integer and store in vector
			line.erase(0, pos + 1); // Erase processed token and comma
		}

		// Process the last token in the line (no comma after it)
		if (!line.empty())
		{
			numbers.push_back(std::stoi(line)); // Convert token to integer and store in vector
		}
	}
	int i = 1;
	std::vector<cv::Rect> boundingBoxes;
	std::vector<int> boundingBox;
	for (auto item : numbers) {
		boundingBox.push_back(item);
		if (i % 4 == 0) {
			cv::Rect rect(boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]);
			boundingBoxes.push_back(rect);
			boundingBox.clear();
		}
		i++;
	}

	std::vector<cv::Rect> finalBoundingBoxes;
	std::vector<bool>merged (boundingBoxes.size(),false);
	// Check if the bounding boxes are intersected somehow, and merge the intersected ones.
	for (int i = 0; i < boundingBoxes.size() - 1; i++) {
		if (merged[i])
			continue;
		for (int j = i + 1;  j < boundingBoxes.size(); j++) {
			if (merged[j])
				continue;
			if (rectanglesIntersect(boundingBoxes[i], boundingBoxes[j]) && (!merged[i]) && (!merged[j])) {
				cv::Rect mergedRect = getBoundingRectangle(boundingBoxes[i], boundingBoxes[j]);
				finalBoundingBoxes.push_back(mergedRect);
				merged[i] = true;
				merged[j] = true;
			}
		}
	}

	for (int i = 0; i < boundingBoxes.size(); i++) {
		if (!merged[i]) {
			finalBoundingBoxes.push_back(boundingBoxes[i]);
		}
	}
	return finalBoundingBoxes;
}


std::vector<std::vector<cv::Rect>> realAllBoundingBoxes(const std::string& path) {
	auto imageBoudingBox = readBoudingBox(path + "//bounding_boxes//food_image_bounding_box.txt");
	auto foodLeftoverBoundingBox1 = readBoudingBox(path + "//bounding_boxes//leftover1_bounding_box.txt");
	auto foodLeftoverBoundingBox2 = readBoudingBox(path + "//bounding_boxes//leftover2_bounding_box.txt");
	auto foodLeftoverBoundingBox3 = readBoudingBox(path + "//bounding_boxes//leftover3_bounding_box.txt");

	std::vector<std::vector<cv::Rect>> result;
	result.push_back(imageBoudingBox);
	result.push_back(foodLeftoverBoundingBox1);
	result.push_back(foodLeftoverBoundingBox2);
	result.push_back(foodLeftoverBoundingBox3);

	return result;
}


void readSegmentationMasks(const std::string& directory, std::vector<cv::Mat>& masks) {

	std::string path = directory + "//masks//";
	cv::Mat foodMask = cv::imread(path + "food_image_mask.png");
	cv::Mat leftoverMask1 = cv::imread(path + "leftover1.png");
	cv::Mat leftoverMask2 = cv::imread(path + "leftover2.png");
	cv::Mat leftoverMask3 = cv::imread(path + "leftover3.png");

	if (foodMask.empty() || leftoverMask1.empty() ||
		leftoverMask2.empty() || leftoverMask3.empty()) {
		std::cerr << "Could not load the image: " << std::endl;
		return;
	}
	masks.push_back(foodMask);
	masks.push_back(leftoverMask1);
	masks.push_back(leftoverMask2);
	masks.push_back(leftoverMask3);
}

std::string selectFolder() {

	// loading and input validation
	std::string trayNumber;
	while (true) {
		std::cout << "Tray number (1 to 8):" << std::endl;
		std::cin >> trayNumber;
		if (isdigit(trayNumber[0]) && stoi(trayNumber) >= 1 && stoi(trayNumber) <= 8) {
			break;
		}
	}
	std::string imagePath = "..//dataset//tray" + trayNumber;
	return imagePath;
}

int main(int argc, char** argv)
{

	std::string imagePath = selectFolder();
	std::vector<cv::Mat> segMasks, images;
	cv::Mat foodImage, foodLeftover1, foodLeftover2, foodLeftover3;
	foodImage = cv::imread(imagePath + "//food_image.jpg", cv::IMREAD_COLOR);
	foodLeftover1 = cv::imread(imagePath + "//leftover1.jpg", cv::IMREAD_COLOR);
	foodLeftover2 = cv::imread(imagePath + "//leftover2.jpg", cv::IMREAD_COLOR);
	foodLeftover3 = cv::imread(imagePath + "//leftover3.jpg", cv::IMREAD_COLOR);
	images.push_back(foodImage);
	images.push_back(foodLeftover1);
	images.push_back(foodLeftover2);
	images.push_back(foodLeftover3);
	readSegmentationMasks(imagePath, segMasks);
	// Check if the images are loaded correctly:
	if (foodImage.empty() || foodLeftover1.empty() ||
		foodLeftover2.empty() || foodLeftover3.empty()) {
		std::cerr << "Could not load the image: " << std::endl;
		return 1;
	}
	
	auto allBoundingBoxes = realAllBoundingBoxes(imagePath);
	std::vector<cv::Rect> foodBBoxCoordinates = allBoundingBoxes[3];
	
	cv::Mat segmentedFoodImage(foodImage.size(),CV_8U);
	std::vector<cv::Mat> segmentationResults;
	for (const auto boundingBox : foodBBoxCoordinates) {
		auto res = GrabcutAlgorithm(foodImage, boundingBox);
		res.copyTo(segmentedFoodImage(boundingBox));
	}
	
	cv::imshow("dest", segmentedFoodImage);
	cv::waitKey(0);
	return 0;
}