#include "foodRecognition.h"

using namespace cv;
using namespace std;

// class for food recognition
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

double featMatching(Mat img1, Mat img2, Mat &Result_MATCHING){
	double ratio = 3; //3;
  cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Ptr<cv::DescriptorMatcher> knn_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  Mat img1_gray, img2_gray, descr1, descr2;

  sift->detectAndCompute(img1, cv::Mat(), keypoints1, descr1);
  sift->detectAndCompute(img2, cv::Mat(), keypoints2, descr2);
	std::vector< std::vector<DMatch> > knn_matches;
	knn_matcher->knnMatch( descr1, descr2, knn_matches, 2 );

	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.8f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
	if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
	{
	good_matches.push_back(knn_matches[i][0]);
	}
	}

  // keeping only good matches (dist less than ratio*min_dist)
  double min_dist = 1000.0;
	std::vector<DMatch> very_good_matches;
	for (int j = 0; j < good_matches.size(); j++)
	{
		double dist = good_matches[j].distance;
		if (dist < min_dist)
			min_dist = dist;
	}
	for (int l = 0; l < good_matches.size(); l++)
	{
		if (good_matches[l].distance <= ratio * min_dist)
		{
			very_good_matches.push_back(good_matches[l]);
		}
	}

	cv::drawMatches(img1, keypoints1, img2, keypoints2, very_good_matches, Result_MATCHING, cv::Scalar::all(-1), cv::Scalar(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return very_good_matches.size();
}

double histMatching(Mat img1, Mat img2){
	Mat hsv_base, hsv_test;
	cvtColor( img1, hsv_base, COLOR_BGR2HSV );
	cvtColor( img2, hsv_test, COLOR_BGR2HSV );
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// Use the 0-th and 1-st channels
	int channels[] = { 0, 1 };
	Mat hist_base, hist_half_down, hist_test;
	calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
	normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
	calcHist( &hsv_test, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false );
	normalize( hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat() );
	double base_test = 0, comp_i = 0;
	std::vector<double> weights{0.1,1.0,0.01,0.1};
	for( int compare_method = 0; compare_method < 4; compare_method++ )
	{
		comp_i = compareHist( hist_base, hist_test, compare_method );
		if(compare_method == 1){
			base_test += weights[compare_method]*1/comp_i;
		}
		else{
			base_test += weights[compare_method]*comp_i;
		}
	}
	return base_test;
}

std::vector<std::tuple<int, int>> boxesMatch(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, std::vector<cv::Rect> boundingBoxes1, std::vector<cv::Rect> boundingBoxes2){
	std::vector<std::tuple<int, int>> matches;
	cv::Rect bb1, bb2;
	cv::Mat segmented1, segmented2, resultFeature;
	img1.copyTo(segmented1, mask1);
	img2.copyTo(segmented2, mask2);

	int maxJ,maxK;
	bool caseSwitch = (boundingBoxes2.size() <= boundingBoxes1.size());
	if(caseSwitch){
		maxJ = boundingBoxes2.size();
		maxK = boundingBoxes1.size();
	}
	else{
		maxJ = boundingBoxes1.size();
		maxK = boundingBoxes2.size();
	}

	std::vector<int> givenCheck(maxK, 0);
	for(int j = 0; j<maxJ; j++){
		//checked bounding box
		if(caseSwitch){
			bb2 = boundingBoxes2[j];
		}
		else{
			bb2 = boundingBoxes1[j];
		}


		double matchFeature, matchHist, totalMatch, bestMatch = 0.0;
		int best_k = 0;
		for( int k = 0;k< maxK; k++){
			//checked leftover bounding box
			if(caseSwitch){
				bb1 = boundingBoxes1[k];
			}
			else{
				bb1 = boundingBoxes2[k];
			}

			//computiong total match
			matchFeature = featMatching(img1(bb1), img2(bb2), resultFeature);
			matchHist = histMatching(img1(bb1), img2(bb2));
			totalMatch = 0.02*matchFeature + 100*matchHist;
			//totalMatch = matchHist;

			//computing best match
			if( totalMatch > bestMatch && givenCheck[k] == 0){
				bestMatch = totalMatch;
				best_k = k;
				givenCheck[k] = 1;
			}
		}
		matches.push_back(std::tuple<int, int>{best_k, j});
	}

	return matches;
};

double foodLeftoverEstimation(const cv::Mat& before, const cv::Mat& after, const cv::Rect& bboxBefore, const cv::Rect& bboxAfter) {

    auto beforeImg = before(bboxBefore);
    auto afterImg = after(bboxAfter);
    double pixelCountBefore = 0, pixelCountAfter = 0;
    for (int y = 0; y < beforeImg.rows; ++y)
    {
        for (int x = 0; x < beforeImg.cols; ++x)
        {
            auto pixel = beforeImg.at<uchar>(y, x);
            if (pixel != 0) {
                pixelCountBefore += 1;
            }
        }
    }

    for (int y = 0; y < afterImg.rows; ++y)
    {
        for (int x = 0; x < afterImg.cols; ++x)
        {
            auto pixel = afterImg.at<uchar>(y, x);
            if (pixel != 0) {
                pixelCountAfter += 1;
            }
        }
    }
    double r = pixelCountAfter / pixelCountBefore;

    return r;
}
