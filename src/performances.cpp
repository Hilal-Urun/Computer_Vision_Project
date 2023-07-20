#include <opencv2/opencv.hpp>

#include "dataHandling.h"
#include "resultsEvaluation.h"

using namespace cv;
using namespace std;

const int N_LEFTOVER_IMAGES = 3;

int main(int argc, char** argv) {
	// Check if the tray n is provided
	if (argc < 2) {
		printf("Warning: tray number not provided\n");
		return -1;
	}
	char* trayN = argv[1];
	string tray_dir = string("data/test/tray") + trayN;
	string results_dir = string("data/results/tray") + trayN;

	std::vector<cv::Mat> groundTruthMasks;
	std::vector<cv::Mat> resultsMasks;
	std::vector<std::vector<cv::Rect>> groundTruthBoundingBoxes;
	std::vector<std::vector<cv::Rect>> resultsBoundingBoxes;

	//Retriving bounding boxes and masks
	for (int i = 0; i < N_LEFTOVER_IMAGES + 1; i++) {
		groundTruthMasks.push_back(readMask(tray_dir, i));
		resultsMasks.push_back(readMask(results_dir, i));
		groundTruthBoundingBoxes.push_back(readBoxes(tray_dir, i));
		resultsBoundingBoxes.push_back(readBoxes(results_dir, i));

	}

	// For food segmentation:
	// The vector size is supposed to be 4 (0 (before), 1(after1), 2(after2), 3(after3)), so we loop on the "after "images
	// of difficulties 1, and 2, i.e: for i = 1, and 2
	vector<double> mIoUres;
	for (int i = 0; i < resultsBoundingBoxes.size(); i++) {
		double result = calculateMeanIoU(groundTruthBoundingBoxes[0], resultsBoundingBoxes[i]);
		mIoUres.push_back(result);
		std::cout << "mIoU of (before) image compared to difficulty image number" << i << " is: " << result << std::endl;
	}
	savemIoU(results_dir,mIoUres);

	// For leftover estimation:
	// The vector size is supposed to be 4 (0 (before), 1(after1), 2(after2), 3(after3)), so we loop on all of the "after "images
	// double result = 0;
	// for (int i = 1; i < resultsBoundingBoxes.size() ; i++) {
	// 	for (int j = 1; j < resultsBoundingBoxes[i].size(); j++) {
	// 		result = foodLeftoverEstimation(resultsMasks[0], resultsMasks[i], resultsBoundingBoxes[0][0], resultsBoundingBoxes[i][j]);
	// 		std::cout << "R_" << j << ": Leftover estimation of food " << j << " in image difficulty: " <<i<< " is: "<< result << std::endl;
	// 	}
	// }
	return 0;
}
