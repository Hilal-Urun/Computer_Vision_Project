#include <opencv2/opencv.hpp>

#include "foodRecognition.h"
#include "dataHandling.h"

using namespace cv;
using namespace std;

const int N_LEFTOVER_IMAGES = 3;
const vector<Scalar> colors{Scalar( 255, 0, 0 ),Scalar( 0, 255, 0 ),Scalar( 0, 0, 255 ),Scalar( 255, 255, 255 ),Scalar( 0, 0, 0 )};

int main(int argc, char **argv){
  // Check if the tray n is provided
  if(argc < 2){
    printf("Warning: tray number not provided\n");
    return -1;
    }
	char* trayN = argv[1];
	string tray_dir = string("data/test/tray")+trayN;
	string results_dir = string("data/results/tray")+trayN;

  // Computing and saving bounding boxes and mask
  vector<Mat> trayImages;
	vector<vector<cv::Rect>> trayBoxes;
  vector<vector<Mat>> trayMasks;
  Mat foodImage, foodMask;
  vector<Rect> foodBoxes;
	for(int i = 0; i<N_LEFTOVER_IMAGES+1; i++){
		//Read food image
		if(i > 0){
      foodImage = imread(tray_dir+"/leftover"+to_string(i)+".jpg");
		}
		else{
			foodImage = imread(tray_dir+"/food_image.jpg");
		}
		if(foodImage.empty()){ // Check if the image files has been introduced correctly
			printf("Warning: image files not provided correctly.\n");
			return -1;
		}
    trayImages.push_back(foodImage);

    //Compute bounding boxes TODO: now just reading the ground truth
    foodBoxes = readBoxes(tray_dir, i);
    trayBoxes.push_back(foodBoxes);
    saveBoxes(results_dir, i, foodBoxes);

    //Compute masks through grab cut algorithm
    vector<Mat> foodMasks;
  	for (int j = 0; j<trayBoxes[trayBoxes.size()-1].size(); j++) {
  		foodMasks.push_back(GrabcutAlgorithm(trayImages[trayImages.size()-1], trayBoxes[trayBoxes.size()-1][j]));
  	}
    trayMasks.push_back(foodMasks);
    saveMasks(results_dir, i, foodMasks, foodImage, foodBoxes);

    //Show the results
    cout << "Image id n: " << i << endl;
    Mat result;
    foodImage.copyTo(result, readMask(results_dir, i));
    namedWindow("Intermediate_result");
		imshow("Intermediate_result", result);
    waitKey(0);
  }

  //Leftover estimation
  Mat loImage, loMask, loSegmented, foodSegmented;
  vector<Rect> loBoxes;
  vector<tuple<int, int>> matches;
  //vector<vector<tuple<int, int>>> overallMatches;
  //vector<double> foodValues;
  for(int k = 0; k<N_LEFTOVER_IMAGES; k++){
    //Gathering data from previous computations
    foodImage = trayImages[k];
    foodBoxes = trayBoxes[k];
    foodMask = readMask(results_dir, k);
    loImage = trayImages[k+1];
    loBoxes = trayBoxes[k+1];
    loMask = readMask(results_dir, k+1);

    //Match bounding boxes
    matches = boxesMatch(foodImage, loImage, foodMask, loMask, foodBoxes, loBoxes);
    //overallMatches.push_back(matches);

    //Compute the leftover values
    vector<double> leftoverValues;
    foodImage.copyTo(foodSegmented, foodMask);
    loImage.copyTo(loSegmented, loMask);
    for(int j = 0; j < matches.size(); j++){
      leftoverValues.push_back(foodLeftoverEstimation(foodSegmented, loSegmented, foodBoxes[get<0>(matches[j])], loBoxes[get<1>(matches[j])]));
    }
    // if(k == 0){
    //   foodValues = leftoverValues;
    // }

    //Show the results
    cout << "Image id n: " << k << endl;
    Mat leftoverComparison, leftImage, rightImage;
    foodImage.copyTo(leftImage);
    loImage.copyTo(rightImage);
    for(int j = 0; j < matches.size(); j++){

      double leftorver_j = leftoverValues[j]; //TODO

      Rect matchedBoxLeft = foodBoxes[get<0>(matches[j])];
      Rect matchedBoxRight = loBoxes[get<1>(matches[j])];
      Point pt1(matchedBoxLeft.x, matchedBoxLeft.y);
      Point pt2(matchedBoxLeft.x+matchedBoxLeft.width, matchedBoxLeft.y+matchedBoxLeft.height);
      rectangle( leftImage, pt1, pt2, colors[j], 2, LINE_8);
      putText( leftImage, to_string(leftoverValues[j]), pt1, FONT_HERSHEY_PLAIN, 2, colors[j] );
      Point pt3(matchedBoxRight.x, matchedBoxRight.y);
      Point pt4(matchedBoxRight.x+matchedBoxRight.width, matchedBoxRight.y+matchedBoxRight.height);
      rectangle( rightImage, pt3, pt4, colors[j], 2, LINE_8);
      putText( rightImage, to_string(leftoverValues[j]), pt3, FONT_HERSHEY_PLAIN, 2, colors[j] );
    }
    try{
      hconcat(leftImage, rightImage, leftoverComparison); // TODO: can fail for mismatching image dimensions
      imshow("Intermediate_result", leftoverComparison);
    }
    catch(exception e){
      namedWindow("Intermediate_result1");
      imshow("Intermediate_result", leftImage);
      imshow("Intermediate_result1", rightImage);
    }
    waitKey(0);
  }

  return 0;
}
