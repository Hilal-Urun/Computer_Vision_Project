#include "foodSegmentation.h"


bool rectanglesIntersect(cv::Rect rect1, cv::Rect rect2) {
	int xOverlap = std::max(0, std::min(rect1.x + rect1.width, rect2.x + rect2.width) - std::max(rect1.x, rect2.x));
	int yOverlap = std::max(0, std::min(rect1.y + rect1.height, rect2.y + rect2.height) - std::max(rect1.y, rect2.y));

	if (xOverlap > 0 && yOverlap > 0) {
		// Rectangles intersect
		return true;
	}
	// Rectangles do not intersect
	return false;
}

cv::Rect getBoundingRectangle(cv::Rect rect1, cv::Rect rect2) {
	int left = std::min(rect1.x, rect2.x);
	int top = std::min(rect1.y, rect2.y);
	int right = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
	int bottom = std::max(rect1.y + rect1.height, rect2.y + rect2.height);

	int width = right - left;
	int height = bottom - top;

	return cv::Rect(left, top, width, height);
}


cv::Mat GrabcutAlgorithm(const cv::Mat& src, const cv::Rect& boundingBox) {

	cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	cv::Mat bgModel, fgModel;

	unsigned int iteration = 5; //Tuneable parameter
	cv::grabCut(src, mask, boundingBox, bgModel, fgModel, iteration, cv::GC_INIT_WITH_RECT);

	cv::Mat mask2 = (mask == 1) + (mask == 3);  // 0 = cv::GC_BGD, 1 = cv::GC_FGD, 2 = cv::PR_BGD, 3 = cv::GC_PR_FGD
	cv::Mat dest;
	src.copyTo(dest, mask2);
	cv::cvtColor(dest, dest, cv::COLOR_RGB2GRAY);
	int newValue = 100; // New value to assign
	int minValue = 50; // Minimum value of the range
	int maxValue = 220; // Maximum value of the range
	for (int y = 0; y < dest.rows; ++y)
	{
		for (int x = 0; x < dest.cols; ++x)
		{
			// Get the pixel value at (x, y)
            auto pixel = dest.at<uchar>(y, x);

            // Check if the pixel value falls within the specified range
            if (pixel >= minValue && pixel <= maxValue) {
                // Set the new pixel value
                pixel = newValue;
            }
			else {
				pixel = 0;
			}

            // Assign the updated pixel value back to the image
			dest.at<uchar>(y, x) = pixel;
		}
	}
	cv::Mat result = dest(boundingBox);

	//cv::imshow("dest", dest);
	return result;
}