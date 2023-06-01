#include <opencv2/opencv.hpp>
#include <iostream>
#include "foodRecognition.h"
#include "foodSegmentation.h"
#include "leftoverEst.h"

using namespace cv;
using namespace std;

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

int main(int argc, char **argv) {
    Mat inputImage = imread("/Users/hilalurun/CLionProjects/Computer_Vision_Project/dataset/tray1/food_image.jpg");
    Mat filteredImage = applyBilateralFilter(inputImage);
    Mat enhancedImage = applyContrastEnhancement(filteredImage);
    Mat colorAdjustedImage = applyColorAdjustment(enhancedImage);

    // Display the original and processed images
    imshow("Original Image", inputImage);
    imshow("Filtered Image", filteredImage);
    imshow("Enhanced Image", enhancedImage);
    imshow("Color Adjusted Image", colorAdjustedImage);
    waitKey(0);

    return 0;
}
