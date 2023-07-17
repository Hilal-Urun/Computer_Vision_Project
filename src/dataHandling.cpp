#include "dataHandling.h"

#include <fstream>

// class for data handling
std::vector<cv::Rect> readBox(char *trayN, int leftoverN) {

	std::ifstream inputFile;
	if(leftoverN != 0){
		inputFile.open(std::string("data/test/tray")+trayN+"/bounding_boxes/leftover"+std::__cxx11::to_string(leftoverN)+"_bounding_box.txt", std::ios::in);
	}
	else{
		inputFile.open(std::string("data/test/tray")+trayN+"/bounding_boxes/food_image_bounding_box.txt", std::ios::in);
	}

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
