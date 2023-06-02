#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "foodRecognition.h"
#include "foodSegmentation.h"
#include "leftoverEst.h"

using namespace cv;
using namespace std;

#define NEIGHBORHOOD_X 9
#define NEIGHBORHOOD_Y 9
#define THRESHOLD Scalar(50,20,20)

//bilateral filter because its smoothing when preserve the edges
Mat applyBilateralFilter(const Mat &inputImage) {
    Mat filteredImage;
    bilateralFilter(inputImage, filteredImage, 9, 75, 75);
    return filteredImage;
}
//
//Mat applyContrastEnhancement(const Mat &inputImage) {
//    Mat enhancedImage;
//    Mat grayImage;
//    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
//    equalizeHist(grayImage, enhancedImage);
//    return enhancedImage;
//}

Mat applyContrastEnhancement(const Mat &filteredImage) {
    Mat enhancedImage;

    // Convert the filtered image to the YCrCb color space
    cvtColor(filteredImage, enhancedImage, COLOR_BGR2YCrCb);

    // Split the YCrCb image into individual channels
    std::vector<Mat> channels;
    split(enhancedImage, channels);

    // Apply histogram equalization to the Y channel
    equalizeHist(channels[0], channels[0]);

    // Merge the enhanced channels back into the YCrCb image
    merge(channels, enhancedImage);

    // Convert the enhanced image back to the BGR color space
    cvtColor(enhancedImage, enhancedImage, COLOR_YCrCb2BGR);

    return enhancedImage;
}


//Hist eq also do same thing, we can ignore it ?

Mat applyColorAdjustment(const Mat &inputImage) {
    Mat adjustedImage;
    // Convert the image to the LAB color space
    cvtColor(inputImage, adjustedImage, COLOR_BGR2Lab);

    // Split the LAB image into L, a, and b channels
    vector<Mat> labChannels(3);
    split(adjustedImage, labChannels);

    // Apply color adjustment on the a and b channels
    labChannels[1] += 10; // Example: increase the a channel by 10
    labChannels[2] -= 10; // Example: decrease the b channel by 10

    // Merge the modified channels back to LAB image
    merge(labChannels, adjustedImage);

    // Convert the LAB image back to the BGR color space
    cvtColor(adjustedImage, adjustedImage, COLOR_Lab2BGR);
    return adjustedImage;
}

void featMatching(Mat img1, Mat img2, Mat &Result_MATCHING){
	double ratio = 1.5; //3;
  cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
  cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
  cv::FlannBasedMatcher Matcher_FLANN;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  vector<cv::DMatch> matches, good_matches;
  cv::Ptr<cv::DescriptorMatcher> Matcher_SIFT = cv::BFMatcher::create(cv::NORM_L2);			// Brute-Force matcher create method
  Mat img1_gray, img2_gray, descr1, descr2;

  /*sift->detectAndCompute(proj1, cv::Mat(), keypoints1, descr1);
  sift->detectAndCompute(proj2, cv::Mat(), keypoints2, descr2);
  Matcher_SIFT->match(descr1, descr2, matches);*/
  fast->detect(img1, keypoints1);
	fast->detect(img2, keypoints2);
	sift->compute(img1, keypoints1, descr1);
	sift->compute(img2, keypoints2, descr2);
  Matcher_FLANN.match(descr1, descr2, matches);

  // keeping only good matches (dist less than ratio*min_dist)
  double max_dist = 0.0, min_dist = 200.0;
  int min_idx;
	for (int j = 0; j < matches.size(); j++)
	{
		double dist = matches[j].distance;
		if (dist < min_dist)
			min_dist = dist;
      min_idx = j;
		if (dist > max_dist)
			max_dist = dist;
	}



  std::vector<cv::Point2f> good_keypoints1, good_keypoints2;
	for (int j = 0; j < matches.size(); j++)
	{
		if (matches[j].distance <= ratio * min_dist)
		{
			good_matches.push_back(matches[j]);
      good_keypoints1.push_back(keypoints1[matches[j].queryIdx].pt);
      good_keypoints2.push_back(keypoints2[matches[j].trainIdx].pt);
		}
	}
  cv::drawKeypoints(img1, keypoints1, Result_MATCHING);
	//cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, Result_MATCHING, cv::Scalar::all(-1), cv::Scalar(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

void onMouse(int event, int x, int y, int f, void* userdata){
  if(event == EVENT_LBUTTONDOWN){
    // Retriving the image from main
    Mat img = *(Mat*) userdata;
    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);

    // Preventing segfaults for looking over the image boundaries
    if (y + NEIGHBORHOOD_Y > img.rows || x + NEIGHBORHOOD_X > img.cols){
      return;
    }

    // Getting the mean of the RGB colors of a 9x9 NEIGHBORHOOD around the selected pixel
    Rect rect(x,y,NEIGHBORHOOD_X,NEIGHBORHOOD_Y);
    Scalar mean_value = mean(img_hsv(rect));
    std::cout << "mean = " << mean_value << std::endl;
    //std::cout << "NEIGHBORHOOD = " << img_out(rect).at<Vec3b> (0,0) << std::endl;

    // Mask computation
    Mat mask0, mask1, mask2;
    inRange(img_hsv, Scalar(mean_value[0]-THRESHOLD[0], mean_value[1], mean_value[2]), Scalar(mean_value[0]+THRESHOLD[0], 255, 255), mask0);
    inRange(img_hsv, Scalar(mean_value[0], mean_value[1]-THRESHOLD[1], mean_value[2]), Scalar(255, mean_value[1]+THRESHOLD[1], 255), mask1);
    inRange(img_hsv, Scalar(mean_value[0], mean_value[1], mean_value[2]-THRESHOLD[2]), Scalar(255, 255, mean_value[2]+THRESHOLD[2]), mask2);
    Mat mask = mask0 | mask1 | mask2;

    // Create and show the window
  	namedWindow("Thresholding mask");
  	imshow("Thresholding mask", mask);
    waitKey(0);
    destroyWindow("Thresholding mask");
    return;
    }
}

int main(int argc, char **argv) {
  // Check if the image file is provided
  if(argc < 2){
    printf("Warning: image file not provided\n");
    return -1;
    }

  // Read the images
  Mat img1 = imread(argv[1]);
  Mat img2 = imread(argv[2]);

  // Check if the image file has been introduced corretly
  if(img1.empty() || img2.empty()){
    printf("Warning: image file not provided correctly. The name or the file type could be wrong.\n");
    return -1;
    }

  // 0. IMAGE PREPROCESSING
  Mat filteredImage1 = applyBilateralFilter(img1);
  Mat enhancedImage1 = applyContrastEnhancement(filteredImage1);
  Mat colorAdjustedImage1 = applyColorAdjustment(enhancedImage1);
  Mat filteredImage2 = applyBilateralFilter(img2);
  Mat enhancedImage2 = applyContrastEnhancement(filteredImage2);
  Mat colorAdjustedImage2 = applyColorAdjustment(enhancedImage2);
  Mat preprocImage11;
  hconcat(img1, filteredImage1, preprocImage11);
  Mat preprocImage12;
  hconcat(enhancedImage1, colorAdjustedImage1, preprocImage12);
  Mat preprocImage;
  vconcat(preprocImage11, preprocImage12, preprocImage);
  namedWindow("Image preprocessing", WINDOW_NORMAL);
  imshow("Image preprocessing", preprocImage);
  resizeWindow("Image preprocessing", Size(600,600));
  waitKey(0);
  //return 0;

  // 1. IMAGE THRESHOLDING
  namedWindow("Image thresholding");
  imshow("Image thresholding", img1);
  setMouseCallback("Image thresholding", onMouse, (void*)&img1);
  waitKey(0);
  //return 0;

  // 2. FEATURE EXTRACTION
  Mat Result_MATCHING;
  featMatching(img1, img2, Result_MATCHING);
  namedWindow("Feature Extraction and Matching");
  imshow("Feature Extraction and Matching", Result_MATCHING);
  waitKey(0);
  return 0;
}
