#include <opencv2/opencv.hpp>

#include "foodRecognition.h"
#include "dataHandling.h"

using namespace cv;
using namespace std;

const int N_LEFTOVER_IMAGES = 3;
const int SHOW_WIDTH  = 900;
const int SHOW_HEIGHT = 600;
const vector<Scalar> colors{Scalar( 255, 0, 0 ),Scalar( 0, 255, 0 ),Scalar( 0, 0, 255 ),Scalar( 255, 255, 255 ),Scalar( 0, 0, 0 )};
const vector<string> colorNames{"Blue","Green", "Red", "White", "Black"};

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
  cout << "Computing and saving bounding boxes and mask..." << endl;
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

    //Reading pre-computed bounding boxes
    foodBoxes = readBoxes(results_dir, i);
    trayBoxes.push_back(foodBoxes);

    //Compute masks through grab cut algorithm
    vector<Mat> foodMasks;
  	for (int j = 0; j<trayBoxes[trayBoxes.size()-1].size(); j++) {
  		foodMasks.push_back(GrabcutAlgorithm(trayImages[trayImages.size()-1], trayBoxes[trayBoxes.size()-1][j]));
  	}
    trayMasks.push_back(foodMasks);
    saveMasks(results_dir, i, foodMasks, foodImage, foodBoxes);

    //Show the results
    Mat result, resultSeg, resultBB;
    foodImage.copyTo(resultBB);
    for(int l = 0; l<foodBoxes.size(); l++){
      Point pt01(foodBoxes[l].x, foodBoxes[l].y);
      Point pt02(foodBoxes[l].x+foodBoxes[l].width, foodBoxes[l].y+foodBoxes[l].height);
      rectangle( resultBB, pt01, pt02, colors[3], 2, LINE_8);
    }
    foodImage.copyTo(resultSeg, readMask(results_dir, i));
    hconcat(resultBB, resultSeg, result);
    resize(result, result, Size(SHOW_WIDTH*2, SHOW_HEIGHT), INTER_LINEAR); //resize down
    namedWindow("Bounding boxes and Segmentation result");
		imshow("Bounding boxes and Segmentation result", result);
    waitKey(0);
  }
  cout << "" << endl;
  destroyWindow("Bounding boxes and Segmentation result");

  //Leftover estimation
  cout << "Leftover estimation..." << endl;
  Mat loImage, loMask, loSegmented, foodSegmented;
  vector<Rect> loBoxes;
  vector<tuple<int, int>> matches;
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

    //Compute the leftover values
    vector<double> leftoverValues;
    foodImage.copyTo(foodSegmented, foodMask);
    loImage.copyTo(loSegmented, loMask);
    for(int j = 0; j < matches.size(); j++){
      leftoverValues.push_back(foodLeftoverEstimation(foodSegmented, loSegmented, foodBoxes[get<0>(matches[j])], loBoxes[get<1>(matches[j])]));
    }

    //Show the results
    Mat leftoverComparison, leftImage, rightImage;
    foodImage.copyTo(leftImage);
    loImage.copyTo(rightImage);
    for(int j = 0; j < matches.size(); j++){
      cout << colorNames[j] << ": " << leftoverValues[j] << endl;
      double leftorver_j = leftoverValues[j];
      Rect matchedBoxLeft = foodBoxes[get<0>(matches[j])];
      Rect matchedBoxRight = loBoxes[get<1>(matches[j])];
      Point pt1(matchedBoxLeft.x, matchedBoxLeft.y);
      Point pt2(matchedBoxLeft.x+matchedBoxLeft.width, matchedBoxLeft.y+matchedBoxLeft.height);
      rectangle( leftImage, pt1, pt2, colors[j], 2, LINE_8);
      Point pt3(matchedBoxRight.x, matchedBoxRight.y);
      Point pt4(matchedBoxRight.x+matchedBoxRight.width, matchedBoxRight.y+matchedBoxRight.height);
      rectangle( rightImage, pt3, pt4, colors[j], 2, LINE_8);
    }
    namedWindow("Leftover estimation");
    try{
      hconcat(leftImage, rightImage, leftoverComparison);
      resize(leftoverComparison, leftoverComparison, Size(SHOW_WIDTH*2, SHOW_HEIGHT), INTER_LINEAR); //resize down
    }
    catch(exception e){
      resize(leftImage, leftImage, Size(SHOW_WIDTH, SHOW_HEIGHT), INTER_LINEAR);
      resize(rightImage, rightImage, Size(SHOW_WIDTH, SHOW_HEIGHT), INTER_LINEAR);
      hconcat(leftImage, rightImage, leftoverComparison);
    }
    Point pt5(leftoverComparison.rows/2-25, leftoverComparison.cols-10);
    Point pt6(leftoverComparison.rows/2+25, leftoverComparison.cols);
    rectangle( leftoverComparison, pt5, pt6, colors[3], FILLED, LINE_8);
    for(int l = 0; l<leftoverValues.size();l++){
        Point pt7(leftoverComparison.rows/2-20, leftoverComparison.cols-8+2*l);
        putText( leftoverComparison, to_string(leftoverValues[l]), pt7, FONT_HERSHEY_PLAIN, 2, colors[l] );
    }
    imshow("Leftover estimation", leftoverComparison);
    waitKey(0);
    cout << "" << endl;
  }

  return 0;
}
