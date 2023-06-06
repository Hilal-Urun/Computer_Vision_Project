#include "foodRecognition.h"

using namespace cv;
using namespace std;

// class for food recognition
FoodRecognition::FoodRecognition(cv::Mat img)
{
	foodImage = img.clone();
}

FoodRecognition::~FoodRecognition()
{
}

Mat FoodRecognition::getFoodImage()
{
	return foodImage;
}

Mat FoodRecognition::runCanny(int lowThreshold, int highThreshold)
{
    cv::cvtColor(foodImage, foodImage, COLOR_BGR2GRAY);
    drawHistogram(foodImage);
    cv::Canny(foodImage, cannyResult, lowThreshold, highThreshold);
	return cannyResult;
}

void FoodRecognition::drawHistogram(cv::Mat img)
{
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    // if RGB image:
    if (img.channels() == 3) {
        std::vector<cv::Mat> bgr_planes;
        split(img, bgr_planes);
        
        cv::Mat b_hist, g_hist, r_hist;
        calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
        calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
        calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
        
        
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; i++)
        {
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                cv::Scalar(255, 0, 0), 2, 8, 0);
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                cv::Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                cv::Scalar(0, 0, 255), 2, 8, 0);
        }
        imshow("Histogram", histImage);
    }
    // Grayscale image
    else {
        std::vector<cv::Mat> grayscale;
        cv::Mat grayHist;
        calcHist(&img, 1, 0, cv::Mat(), grayHist, 1, &histSize, histRange, uniform, accumulate);
        cv::Mat histImage(hist_h, hist_w, CV_8U, cv::Scalar(0, 0, 0));
        normalize(grayHist, grayHist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());
        for (int i = 1; i < histSize; i++)
        {
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(grayHist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(grayHist.at<float>(i))),
                cv::Scalar(255), 2, 8, 0);
        }
        imshow("Histogram", histImage);
    }
}
