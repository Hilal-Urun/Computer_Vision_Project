#include "foodRecognition.h"

using namespace cv;
using namespace std;

// class for food recognition
double featMatching(Mat img1, Mat img2, Mat &Result_MATCHING){
	double ratio = 3; //3;
  cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
  cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
  cv::FlannBasedMatcher Matcher_FLANN;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  //vector<cv::DMatch> matches, good_matches;
	cv::Ptr<cv::DescriptorMatcher> knn_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  cv::Ptr<cv::DescriptorMatcher> Matcher_SIFT = cv::BFMatcher::create(cv::NORM_L2);			// Brute-Force matcher create method
  Mat img1_gray, img2_gray, descr1, descr2;

  sift->detectAndCompute(img1, cv::Mat(), keypoints1, descr1);
  sift->detectAndCompute(img2, cv::Mat(), keypoints2, descr2);
  //Matcher_SIFT->match(descr1, descr2, matches);
	std::vector< std::vector<DMatch> > knn_matches;
	knn_matcher->knnMatch( descr1, descr2, knn_matches, 2 );
  /*fast->detect(img1, keypoints1);
	fast->detect(img2, keypoints2);
	sift->compute(img1, keypoints1, descr1);
	sift->compute(img2, keypoints2, descr2);*/
  //Matcher_FLANN.match(descr1, descr2, matches);

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

double histMatching(Mat img1, Mat img2, Mat &Result_MATCHING){
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
			//cout << weights[compare_method]*1/comp_i << endl;
		}
		else{
			base_test += weights[compare_method]*comp_i;
			//cout << weights[compare_method]*comp_i << endl;
		}
	}
	//cout << base_test << endl;
	return base_test;
}
