#ifndef DATA_HANDLING_H
#define DATA_HANDLING_H
#endif

#include <opencv2/opencv.hpp>

// functions for data handling
std::vector<cv::Rect> readBoxes(std::string tra_dir, int leftoverN);
void saveBoxes(std::string results_dir, int leftoverN, std::vector<cv::Rect> boundingBoxes);
cv::Mat readMask(std::string tray_dir, int leftoverN);
void saveMasks(std::string results_dir, int leftoverN, std::vector<cv::Mat> masks, cv::Mat originalImage, std::vector<cv::Rect> boundingBoxes);
void savemIoU(std::string results_dir, std::vector<double> mIoU);
