// To store the world points which are the corners of the chess board
// To store the corresponding image points of the corners
// Then we need to find a relation between the world points and image points
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <iterator>
#include <algorithm>
#include <glob.h>

using namespace std;
using namespace cv;


int main()
{
    int count = 0;
    int success = 0;
    int numBoards = 9;
    int numCornersHor = 8;
    int numCornersVer = 5;
    float square_size = 24.23;
    int board_height = numCornersVer*square_size;
    int board_width = numCornersHor*square_size;
    Mat img, gray;
    //        cout << "Enter number of horizontal corners: \t";
    //        cin >> numCornersHor;

    //        cout << "Enter number of vertical corners: \t";
    //        cin >> numCornersVer;

    //        cout << "Enter number of board photos: \t";
    //        cin >> numBoards;

    //        cout << "Number of corner points:\t" << numOfPoints;
    //        cout << "\nSize of board: \t" << board_size <<"\n";

    int numOfPoints = numCornersHor*numCornersVer;
    Size boardSize = Size(numCornersHor,numCornersVer);

    /* DATA POINTS  */
    vector<vector<Point3f> > world_points;  // To store the world co-ordinates
    vector<vector<Point2f> > image_points;  // To store the image co-ordinates

    /*  DEFINING THE WORLD CO-ORDINATES */
    vector< Point3f > objects;
    for (int i = 0; i < numCornersVer; i++)
        for (int j = 0; j < numCornersHor; j++)
            objects.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));
    cout <<"\nThe world coordinates are at: \n" <<objects<< "\n";
    cout <<"\nTotal number of points in world: \t" <<objects.size();

    /*  DETERMINING THE POSITION OF CORNERS IN IMAGE FRAME  */
    while(success < numBoards)
    {
        cv::String  folder = "/home/ragesh/C++ /calibration_laptop/images/*.jpg";
        std::vector<cv::String> filename;

        glob(folder, filename, false);

        //glob(folder.c_str(), filename, false);
        if(count < filename.size())
        {

            img = imread(filename[count]);
            if(img.empty())
            {
                cout << "\nimage loading failed.....!"<<"\n";
                return 0;
            }
            //imshow("image", img);

        }
        cvtColor ( img,gray, COLOR_BGR2GRAY );// gray scale the source image
        vector<Point2f> corners; //this will be filled by the detected corners

        bool patternfound = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK); //CALIB_CB_FAST_CHECK saves a lot of time on images that do not contain any chessboard corners
        if(patternfound)
        {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(img, boardSize, Mat(corners), patternfound);
            cout<< "\n\nCorners detected at: \n\n" << corners << "\nTotal  nuber of corners detected:\t" <<corners.size() << "\n";
            imshow("result", img);
            waitKey(0);
            image_points.push_back(corners);
            world_points.push_back(objects);
            success++;
        }
        count++;
    }
    /*  CALIBRATION OF CAMERA*/
    cout << "\n started calibration of camera........\n";
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    calibrateCamera(world_points, image_points, img.size(), intrinsic, distCoeffs, rvecs, tvecs);

    cout << "\n\n intrinsic:-\n" << intrinsic;
    cout << "\n\n distCoeffs:-\n" << distCoeffs;
    copy(rvecs.begin(), rvecs.end(), ostream_iterator<Mat>(cout, "\n\n Rotation vector:-\n "));
    copy(tvecs.begin(), tvecs.end(), ostream_iterator<Mat>(cout, "\n\n Translational vector:-\n"));
    cout << "\n board_width:\t" << board_width;
    cout << "\n board_height:\t" << board_height;
    cout << "\n square_size:\t" << square_size;
    printf("\n\nDone Calibration.........!\n\n");

    /*  REPROJECTION ERROR  */
    vector< Point2f > imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    vector< float > perViewErrors;
    perViewErrors.resize(world_points.size());
    for (i = 0; i < (int)world_points.size(); ++i)
    {
        projectPoints(Mat(world_points[i]), rvecs[i], tvecs[i], intrinsic, distCoeffs, imagePoints2);
        err = norm(Mat(image_points[i]), Mat(imagePoints2), CV_L2);
        int n = (int)world_points[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }
    double reproj_error = sqrt(totalErr/totalPoints);
    cout << "\n\nReprojection error: \t\t" << reproj_error;
    cout << "\n\n\n";
    return 0;

}
