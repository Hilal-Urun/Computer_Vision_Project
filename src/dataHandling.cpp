#include "dataHandling.h"

#include <fstream>

// function for reading bounding boxes
std::vector<cv::Rect> readBoxes(std::string tray_dir, int leftoverN) {
	std::ifstream inputFile;
	if(leftoverN != 0){
		inputFile.open(tray_dir+"/bounding_boxes/leftover"+std::__cxx11::to_string(leftoverN)+"_bounding_box.txt", std::ios::in);
	}
	else{
		inputFile.open(tray_dir+"/bounding_boxes/food_image_bounding_box.txt", std::ios::in);
	}

	if (!inputFile)
	{
		std::cout << "Failed to read the bounding boxes file." << std::endl;
		std::vector<cv::Rect> emptyVector;
		return emptyVector;
	}

	std::vector<int> numbers;
	std::string line;
	while (std::getline(inputFile, line))
	{
		std::string token;
		size_t pos = 0;
		line.erase(0, line.find('[') + 1);
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

	return boundingBoxes;
}

// function for saving bounding boxes
void saveBoxes(std::string results_dir, int leftoverN, std::vector<cv::Rect> boundingBoxes) {
	std::ofstream outputFile;
	if(leftoverN != 0){
		outputFile.open(results_dir+"/bounding_boxes/leftover"+std::__cxx11::to_string(leftoverN)+"_bounding_box.txt");
	}
	else{
		outputFile.open(results_dir+"/bounding_boxes/food_image_bounding_box.txt");
	}

	if (!outputFile)
	{
		std::cout << "Failed to save the bounding boxes file." << std::endl;
	}

	for(int i = 0; i < boundingBoxes.size(); i++){
		outputFile << "[";
		std::vector<int> boundingBox{boundingBoxes[i].x,boundingBoxes[i].y,boundingBoxes[i].width,boundingBoxes[i].height};
		for(int j = 0; j < 4; j++){
			if (j == 3){
				outputFile << std::__cxx11::to_string(boundingBox[j]);
			}
			else{
				outputFile << std::__cxx11::to_string(boundingBox[j])+", ";
			}
		}
		outputFile << "]\n";
	}
	outputFile.close();
}

// function for reading masks
cv::Mat readMask(std::string tray_dir, int leftoverN) {
	std::string imgPath;
	if(leftoverN != 0){
		imgPath = tray_dir+"/masks/leftover"+std::__cxx11::to_string(leftoverN)+".png";
	}
	else{
		imgPath = tray_dir+"/masks/food_image_mask.png";
	}

	cv::Mat mask = cv::imread(imgPath);
	if(mask.empty()){
		std::cout << "Failed to read the mask." << std::endl;
	}
	return mask;
}

// function for saving masks
void saveMasks(std::string results_dir, int leftoverN, std::vector<cv::Mat> masks, cv::Mat originalImage, std::vector<cv::Rect> boundingBoxes) {
	std::string imgPath;
	if(leftoverN != 0){
		imgPath = results_dir+"/masks/leftover"+std::__cxx11::to_string(leftoverN)+".png";
	}
	else{
		imgPath = results_dir+"/masks/food_image_mask.png";
	}

	cv::Mat overallMask(originalImage.size(),CV_8U);
	for(int i = 0; i<masks.size(); i++){
		masks[i].copyTo(overallMask(boundingBoxes[i]));
	}

	bool check = cv::imwrite(imgPath, overallMask);
	if(!check){
		std::cout << "Failed to save the masks." << std::endl;
	}
}
