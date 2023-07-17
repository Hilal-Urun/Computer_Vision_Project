#include "foodRecognition.h"


double calculateIoU(const cv::Rect& bbox1, const cv::Rect& groundTruth) {
    // Calculate the intersection coordinates
    int x1 = std::max(bbox1.x, groundTruth.x);
    int y1 = std::max(bbox1.y, groundTruth.y);
    int x2 = std::min(bbox1.x + bbox1.width, groundTruth.x + groundTruth.width);
    int y2 = std::min(bbox1.y + bbox1.height, groundTruth.y + groundTruth.height);

    // Calculate the intersection area
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);

    // Calculate the union area
    int bbox1Area = bbox1.width * bbox1.height;
    int bbox2Area = groundTruth.width * groundTruth.height;
    int unionArea = bbox1Area + bbox2Area - intersectionArea;

    // Calculate the IoU
    double iou = static_cast<double>(intersectionArea) / static_cast<double>(unionArea);

    return iou;
}



double calculateMeanIoU(const std::vector<cv::Rect>& gtBboxes, const std::vector<cv::Rect>& predBboxes) {
    // Calculate IoU for each pair of ground truth and predicted bounding boxes
    std::vector<double> ious;

    for (const cv::Rect& gtBbox : gtBboxes) {
        double maxIoU = 0.0;
        for (const cv::Rect& predBbox : predBboxes) {
            double iou = calculateIoU(gtBbox, predBbox);
            maxIoU = std::max(maxIoU, iou);
        }
        ious.push_back(maxIoU);
    }

    // Calculate the mean IoU
    double meanIoU = 0.0;

    if (!ious.empty()) {
        double sumIoU = 0.0;

        for (double iou : ious) {
            sumIoU += iou;
        }

        meanIoU = sumIoU / static_cast<double>(ious.size());
    }
    return meanIoU;
}

/*
double calculateAP(const std::vector<cv::Rect>& gtBboxes, const std::vector<cv::Rect>& predBboxes, double iouThreshold) {
    // Sort the predicted bounding boxes by confidence score in descending order
    std::vector<cv::Rect> sortedPredBboxes = predBboxes;
    std::sort(sortedPredBboxes.begin(), sortedPredBboxes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.confidence > b.confidence;
        });

    // Initialize variables for calculating precision and recall
    int numTruePositives = 0;
    int numFalsePositives = 0;

    // Initialize variables for calculating average precision
    double cumulativePrecision = 0.0;
    double cumulativeRecall = 0.0;
    int numGroundTruth = gtBboxes.size();

    // Iterate through the predicted bounding boxes
    for (const cv::Rect& predBbox : sortedPredBboxes) {
        // Find the ground truth bounding box with the highest IoU
        double maxIoU = 0.0;
        int maxIoUIdx = -1;

        for (size_t i = 0; i < gtBboxes.size(); ++i) {
            double iou = calculateIoU(gtBboxes[i], predBbox);

            if (iou > maxIoU) {
                maxIoU = iou;
                maxIoUIdx = i;
            }
        }

        // Check if the highest IoU exceeds the threshold
        if (maxIoU >= iouThreshold) {
            // Check if the ground truth bounding box is not already matched
            if (!gtBboxes[maxIoUIdx].matched) {
                // Increase the true positive count
                ++numTruePositives;

                // Mark the ground truth bounding box as matched
                gtBboxes[maxIoUIdx].matched = true;
            }
            else {
                // Increase the false positive count
                ++numFalsePositives;
            }
        }
        else {
            // Increase the false positive count
            ++numFalsePositives;
        }

        // Calculate precision and recall at this step
        double precision = static_cast<double>(numTruePositives) / static_cast<double>(numTruePositives + numFalsePositives);
        double recall = static_cast<double>(numTruePositives) / static_cast<double>(numGroundTruth);

        // Accumulate precision and recall for calculating average precision
        cumulativePrecision += precision;
        cumulativeRecall += recall;
    }

    // Calculate average precision
    double averagePrecision = cumulativePrecision / static_cast<double>(sortedPredBboxes.size());

    return averagePrecision;
}

*/
double meanAveragePrecision() {




	return 0 ;
}


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