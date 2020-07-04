#include <vector>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static float *params; 

 int main()
 {
    Mat  img = imread(img_path,0);
    pparams = new float[img1.rows*img1.cols*8];
	std::ifstream ifs("the path of param txt", std::ios::binary | std::ios::in);
	ifs.read((char*)pparams, sizeof(float) * img1.rows*img1.cols*8);

//the structure of the p1p2 param if setted mannally can be:
//      		for(int i=0; i< width*height*8 ;i +=8 )
//	    {
//	        /// down to up
//	        pparams[i] = 5;
//	        pparams[i+1] = 80.0;
//	        /// left to right 
//	        pparams[i+2] = 5.0;
//	        pparams[i+3] = 80.0;
//	        /// up to down 
//	        pparams[i+4] = 5.0;
//	        pparams[i+5] = 80.0;
//	        /// right to left
//	        pparams[i+6] = 5.0;
//	        pparams[i+7] = 80.0;
//	    }
