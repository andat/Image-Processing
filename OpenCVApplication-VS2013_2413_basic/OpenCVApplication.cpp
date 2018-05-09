// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <queue>
#include <random>
#include <fstream>

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}


void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		imshow("source", frame);
		imshow("gray", grayFrame);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

unsigned int ADDITIVE_FACTOR = 50;
void change_gray_additive() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("before", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) + ADDITIVE_FACTOR > 255)
				img.at<uchar>(i, j)= 255;
			else if(img.at<uchar>(i, j) + ADDITIVE_FACTOR < 0)
				img.at<uchar>(i, j) = 0;
			else
				img.at<uchar>(i, j) += ADDITIVE_FACTOR;
		}
	}
	imshow("after addition", img);
	waitKey(0);
}

unsigned int MULTIPLICATIVE_FACTOR = 2;
void change_gray_multiplicative() {
	Mat img = imread("Images/saturn.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("before", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) * MULTIPLICATIVE_FACTOR > 255)
				img.at<uchar>(i, j) = 255;
			else if (img.at<uchar>(i, j) * MULTIPLICATIVE_FACTOR < 0)
				img.at<uchar>(i, j) = 0;
			else
				img.at<uchar>(i, j) *= MULTIPLICATIVE_FACTOR;
		}
	}
	imwrite("ProcessedImages/saturn.bmp", img);
	imshow("after multiplication", img);
	waitKey(0);
}

void create_color_image(){
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			Vec3b pixel;
			if (i < 128 && j < 128) {
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			else if (i < 128 && j >= 128) {
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			else if (i >= 128 && j < 128) {
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
			}
			else  {
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 0;
			}
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	imshow("squares", img);
	waitKey(0);
}

void horizontal_flip() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat copy(img.rows, img.cols, CV_8UC1);
	imshow("before", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			copy.at<uchar>(i, j) = img.at<uchar>(img.rows - i-1, j);
		}
	}
	imshow("after", copy);
	waitKey(0);
}

void vertical_flip() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat copy(img.rows, img.cols, CV_8UC1);
	imshow("before", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			copy.at<uchar>(i, j) = img.at<uchar>(i, img.rows - j - 1);
		}
	}
	imshow("after", copy);
	waitKey(0);
}

unsigned int height = 100;
unsigned int width = 100;
void center_crop() {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", img);
	Mat cropped(height, width, CV_8UC3);
	int starti = (img.rows - height) / 2;
	int startj = (img.cols - width) / 2;
	for (int i = starti; i < img.rows - starti - 1; i++) {
		for (int j = startj; j < img.cols - startj - 1; j++) {
			cropped.at<Vec3b>(i - starti, j - startj) = img.at<Vec3b>(i, j);
		}
	}
	imshow("cropped", cropped);
	waitKey(0);
}

void resize() {
	int new_height, new_width;
	printf("Enter new height:\n");
	std::cin >> new_height;
	printf("Enter new width:\n");
	std::cin >> new_width;

	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", img);
	Mat aux(new_height, new_width, CV_8UC3);
	float ratioHeight = (float) img.rows / new_height;
	float  ratioWidth = (float) img.cols / new_width;
	for (int i = 0; i < new_height; i++) {
		for (int j = 0; j < new_width; j++) {
			int oldi = (int)i * ratioHeight;
			int oldj = (int)j * ratioWidth;
			aux.at<Vec3b>(i, j) = img.at<Vec3b>(oldi, oldj);
		}
	}
	imshow("resized image", aux);
	waitKey(0);
}

void rgb_to_three() {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", img);
	Mat rcopy(img.rows, img.cols, CV_8UC1);
	Mat gcopy(img.rows, img.cols, CV_8UC1);
	Mat bcopy(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i,j);
			rcopy.at<uchar>(i, j) = pixel[2];
			gcopy.at<uchar>(i, j) = pixel[1];
			bcopy.at<uchar>(i, j) = pixel[0];
		}
	}

	imshow("rcopy", rcopy);
	imshow("gcopy", gcopy);
	imshow("bcopy", bcopy);
	waitKey(0);
}

void rgb_to_grayscale() {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", img);
	Mat copy(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			copy.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}
	}
	imshow("grayscale", copy);
	waitKey(0);
}

void grayscale_to_binary() {
	unsigned int threshold;
	std::cout << "Enter threshold: ";
	std::cin >> threshold;

	//Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("before", img);
	Mat copy(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < threshold)
				copy.at<uchar>(i, j) = 0;
			else 
				copy.at<uchar>(i, j) = 255;
		}
	}

	imshow("binary image", copy);
	waitKey(0);
}

void rgb_to_hsv() {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", img);

	Mat hcopy(img.rows, img.cols, CV_8UC1);
	Mat scopy(img.rows, img.cols, CV_8UC1);
	Mat vcopy(img.rows, img.cols, CV_8UC1);

	float r, g, b, s, h;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			r = (float)pixel[2]/255;
			g = (float)pixel[1]/255;
			b = (float)pixel[0]/255;

			float m = max(max(r, g), b);
			float c = m - min(min(r, g), b);

			//value
			float v = m;
			//saturation
			if (v != 0)
				s = c / v;
			else
				s = 0;
			//hue
			if (c != 0.0) {
				if (m == r)
					h = 60 * (g - b) / c;
				if (m == g)
					h = 120 + 60 * (b - r) / c;
				if (m == b)
					h = 240 + 60 * (r - g) / c;

			}
			else
				h = 0;

			if (h < 0)
				h = h + 360;

			hcopy.at<uchar>(i, j) = h * 255 / 360;
			scopy.at<uchar>(i, j) = s * 255;
			vcopy.at<uchar>(i, j) = v * 255;
		}
	}

	imshow("hue", hcopy);
	imshow("saturation", scopy);
	imshow("value", vcopy);
	waitKey(0);
}

int compute_area(Mat* img, Vec3b colour) {
	int area = 0;
	for (int i = 0; i < img->rows; i++) {
		for (int j = 0; j < img->cols; j++) {
			if (img->at<Vec3b>(i, j) == colour)
				area++;
		}
	}
	return area;
}

int* compute_center_of_mass(Mat* img, Vec3b colour, int area, Mat copy) {
	int center[2];
	int rowsum = 0;
	int colsum = 0;
	for (int i = 0; i < img->rows; i++) {
		for (int j = 0; j < img->cols; j++) {
			if (img->at<Vec3b>(i, j) == colour) {
				rowsum += i;
				colsum += j;
			}
		}
	}
	center[0] = rowsum / area;
	center[1] = colsum / area;
	copy.at<Vec3b>(center[0], center[1]) = Vec3b(0, 0, 0);
	return center;
}

double compute_axis_of_elongation(Mat* img, Vec3b colour, int r_center, int c_center) {
	double nominator = 0;
	double denominator = 0;
	/*int r_center = center[0];
	int c_center = center[1];
*/
	for (int i = 0; i < img->rows; i++) {
		for (int j = 0; j < img->cols; j++) {
			if (img->at<Vec3b>(i, j) == colour) {
				nominator += (double) (i - r_center) * (j - c_center);
				denominator += pow((j - c_center), 2) - pow((i - r_center), 2);
			}
		}
	}
	double angle = (atan2(2 * nominator, denominator) / 2.0)* (180.0 / PI);
	if (angle < 0)
		angle += 180;
	return angle;
}

bool is_white(Vec3b pixel) {
	if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
		return true;
	else 
		return false;
}

int compute_perimeter(Mat* img, Vec3b colour, Mat copy) {
	int perimeter = 0;
	for (int i = 1; i < img->rows-1; i++) {
		for (int j = 1; j < img->cols-1; j++) {
			if (img->at<Vec3b>(i, j) == colour) {
				if (is_white(img->at<Vec3b>(i - 1, j)) || is_white(img->at<Vec3b>(i - 1, j - 1)) || is_white(img->at<Vec3b>(i - 1, j + 1)) || is_white(img->at<Vec3b>(i, j - 1))
					|| is_white(img->at<Vec3b>(i, j + 1)) || is_white(img->at<Vec3b>(i + 1, j - 1)) || is_white(img->at<Vec3b>(i + 1, j + 1)) || is_white(img->at<Vec3b>(i + 1, j))) {
					perimeter++;
					copy.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
			}
		}
	}
	return perimeter * (PI / 4);
}

float compute_thinness_ratio(Mat* img, Vec3b colour, int area, int perimeter) {
	float ratio = area / pow(perimeter, 2);
	return 4 * PI * ratio;
}

float compute_aspect_ratio(Mat* img, Vec3b colour) {
	int cmax = 0;
	int cmin = img->cols;
	int rmax = 0;
	int rmin = img->rows;

	for (int i = 1; i < img->rows - 1; i++) {
		for (int j = 1; j < img->cols - 1; j++) {
			if (img->at<Vec3b>(i, j) == colour) {
				if (j > cmax)
					cmax = j;
				if (j < cmin)
					cmin = j;
				if (i > rmax)
					rmax = i;
				if (i < rmin)
					rmin = i;
			}
		}
	}
	float ratio = ((float) (cmax - cmin + 1)) / (rmax - rmin + 1);
	return ratio;
}


void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	Mat copy = src->clone();
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		int area = compute_area(src, (*src).at<Vec3b>(y, x));
		printf("\nArea is: %d\n", area);
		int* center = compute_center_of_mass(src, (*src).at<Vec3b>(y, x), area, copy);
		int r_center = center[0];
		int c_center = center[1];

		printf("Center of mass is at: row %d column %d\n", center[0], center[1]);
		double angle = compute_axis_of_elongation(src, (*src).at<Vec3b>(y, x), r_center, c_center);
		printf("Angle of the axis of elongation: %lf\n", angle);
		//draw line
		int length = 70;
		int dx = sin(angle * PI / 180.0) * length;
		int dy = cos(angle * PI / 180.0) * length;
		line(copy, Point(c_center - dy, r_center - dx), Point(c_center + dy, r_center + dx), 1, 1);

		int perimeter = compute_perimeter(src, (*src).at<Vec3b>(y, x), copy);
		printf("Perimeter: %d\n", perimeter);
		printf("Thinness ratio: %.2f\n", compute_thinness_ratio(src, (*src).at<Vec3b>(y, x), area, perimeter));
		printf("Aspect ratio: %.2f\n", compute_aspect_ratio(src, (*src).at<Vec3b>(y, x)));
		imshow("Copy", copy);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void select_object() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void detect_traffic_sign() {
	Mat bgrImg = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("before", bgrImg);
	Mat hsvImg;
	cv::cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

	Mat hsv_channels[3];
	cv::split(hsvImg, hsv_channels);
	Mat hue = hsv_channels[0];

	Mat mask;
	cv::inRange(hsv_channels[0], 0, 7, mask);
	
	//imshow("hsv", hsvImg);
	imshow("mask", mask);
	waitKey(0);
}

vector<Point2i> getNeighbours8(int height, int width, Point2i el) {
	vector<Point2i> neighbours;
	int i = el.x;
	int j = el.y;
	
	neighbours.push_back(Point2i(i - 1, j));
	neighbours.push_back(Point2i(i - 1, j - 1));
	neighbours.push_back(Point2i(i - 1, j + 1));
	neighbours.push_back(Point2i(i, j - 1));
	neighbours.push_back(Point2i(i, j + 1));
	neighbours.push_back(Point2i(i + 1, j - 1));
	neighbours.push_back(Point2i(i + 1, j));
	neighbours.push_back(Point2i(i + 1, j + 1));

	int index = 0;
	for (Point2i p : neighbours) {
		if (p.x >= height || p.x < 0 || p.y >= width || p.y < 0) {
			neighbours.erase(neighbours.begin() + index);
		}
		index++;
	}

	return neighbours;
}

vector<Point2i> get_prev_neigbours(int height, int width, Point2i el) {
	vector<Point2i> neighbours;
	int i = el.x;
	int j = el.y;

	neighbours.push_back(Point2i(i - 1, j));
	neighbours.push_back(Point2i(i - 1, j - 1));
	neighbours.push_back(Point2i(i - 1, j + 1));
	neighbours.push_back(Point2i(i, j - 1));

	int index = 0;
	for (Point2i p : neighbours) {
		if (p.x >= height || p.x < 0 || p.y >= width || p.y < 0) {
			neighbours.erase(neighbours.begin() + index);
		}
		index++;
	}

	return neighbours;
}

void colour(Mat labels, Mat coloured) {
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	Vec3b colors[50] = { Vec3b(0,0,0) };

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels.at<uchar>(i, j) == 0) {
				coloured.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if(colors[labels.at<uchar>(i, j)] != Vec3b(0, 0, 0))
				coloured.at<Vec3b>(i, j) = colors[labels.at<uchar>(i, j)];
			else {
				uchar r = d(gen);
				uchar g = d(gen);
				uchar b = d(gen);
				coloured.at<Vec3b>(i, j) = Vec3b(r, g, b);
				colors[labels.at<uchar>(i, j)] = Vec3b(r, g, b);
			}
		}
	}
}

void label_BFS() {
	Mat img = imread("Images/labeling1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	int label = 0;
	Mat labels(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels.at<uchar>(i, j) = 0;
		}
	}
	Mat coloured(img.rows, img.cols, CV_8UC3);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
				label++;
				std::queue<Point2i> q;
				labels.at<uchar>(i, j) = label;
				q.push(Point2i(i, j));
				while (!q.empty()) {
					Point2i el = (Point2i)q.front();
					q.pop();
					vector<Point2i> neigh = getNeighbours8(img.rows, img.cols, el);
					for (Point2i n : neigh) {
						int in = n.x;
						int jn = n.y;
						if (img.at<uchar>(in, jn) == 0 && labels.at<uchar>(in, jn) == 0) {
							labels.at<uchar>(in, jn) = label;
							q.push(n);
						}
					}
				}
			}
		}
	}

	colour(labels, coloured);
	//imshow("labelled image", labels);
	imshow("labelled image", coloured);
	waitKey(0);
}

void two_pass() {
	Mat img = imread("Images/labeling1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	int label = 0;
	Mat labels(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels.at<uchar>(i, j) = 0;
		}
	}

	std::vector<std::vector<int>> edges(100, std::vector<int>(100));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
				vector<int> L;
				vector<Point2i> neigh = get_prev_neigbours(img.rows, img.cols, Point2i(i, j));
				for (Point2i n : neigh) {
					if (labels.at<uchar>(n.x, n.y) > 0)
						L.push_back(labels.at<uchar>(n.x, n.y));
				}
				if (L.size() == 0) {
					label++;
					labels.at<uchar>(i, j) = label;
				} else {
					 int x = min_element(L.begin(), L.end())[0];
					 labels.at<uchar>(i, j) = x;
					 for (int y : L) {
						 if (y != x) {
							 edges[x].push_back(y);
							 edges[y].push_back(x);
						 }
					 }
				}
			}
		}
	}	
	imshow("intermediate", labels);

	int newlabel = 0;
	vector<int> newlabels(label + 1);
	for (int i = 1; i < label; i++) {
		if (newlabels.at(i) == 0) {
			newlabel++;
			std::queue<int> q;
			newlabels.at(i) = newlabel;
			q.push(i);
			while (!q.empty()) {
				int x = q.front();
				q.pop();
				for (int y : edges[x]) {
					if (newlabels.at(y) == 0) {
						newlabels.at(y) = newlabel;
						q.push(y);
					}
				}
			}
		}
	}

	for(int i = 0; i < labels.rows; i++)
		for (int j = 0; j < labels.cols; j++) {
			labels.at<uchar>(i, j) = newlabels[labels.at<uchar>(i, j)];
		}

	Mat coloured(img.rows, img.cols, CV_8UC3);
	colour(labels, coloured);
	//imshow("labelled image", labels);
	imshow("labelled image", coloured);
	waitKey(0);
}

Point2i getStartPoint(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0)
				return Point2i(i, j);
		}
	}
}

Point2i getPixelAtDir(int dir, Point2i p) {
	int i = p.x;
	int j = p.y;
	switch (dir) {
		case 0:
			return Point2i(i, j + 1);
		case 1:
			return Point2i(i - 1, j + 1);
		case 2:
			return Point2i(i - 1, j);
		case 3:
			return Point2i(i - 1, j - 1);
		case 4:
			return Point2i(i, j - 1);
		case 5:
			return Point2i(i + 1, j - 1);
		case 6:
			return Point2i(i + 1, j);
		case 7:
			return Point2i(i + 1, j + 1);

	}

}

void trace_border() {
	Mat img = imread("Images/triangle_up.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	Mat copy(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			copy.at<uchar>(i, j) = 255;
		}
	}

	Point2i startPoint = getStartPoint(img);
	Point2i currPoint = startPoint;
	Point2i nextPoint;
	vector<Point2i> border;
	int n = border.size();
	int dir = 7;
	border.push_back(startPoint);

	do {
		if (dir % 2 == 0)
			dir = (dir + 7) % 8;
		else
			dir = (dir + 6) % 8;

		nextPoint = getPixelAtDir(dir, currPoint);
		while (img.at<uchar>(nextPoint.x, nextPoint.y) == 255) {
				dir = (dir + 1) % 8;
				nextPoint = getPixelAtDir(dir, currPoint);
		}

		border.push_back(nextPoint);
		currPoint = nextPoint;

		n = border.size();
	} while (n <= 2 || (border.at(0) != border.at(n- 2) && border.at(1) != border.at(n-1)) );
	
	//colour copy image
	for (Point2i p : border) {
		copy.at<uchar>(p.x, p.y) = 0;
	}
	imshow("border", copy);
	waitKey(0);
}

void chain_code() {
	Mat img = imread("Images/triangle_up.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	Mat copy(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			copy.at<uchar>(i, j) = 255;
		}
	}

	Point2i startPoint = getStartPoint(img);
	Point2i currPoint = startPoint;
	Point2i nextPoint;
	vector<Point2i> border;
	vector<int> code;
	vector<int> derivative;

	int n = border.size();
	int dir = 7;
	border.push_back(startPoint);

	do {
		if (dir % 2 == 0)
			dir = (dir + 7) % 8;
		else
			dir = (dir + 6) % 8;

		nextPoint = getPixelAtDir(dir, currPoint);
		while (img.at<uchar>(nextPoint.x, nextPoint.y) == 255) {
			dir = (dir + 1) % 8;
			nextPoint = getPixelAtDir(dir, currPoint);
		}

		border.push_back(nextPoint);
		code.push_back(dir);

		currPoint = nextPoint;

		n = border.size();
	} while (n <= 2 || (border.at(0) != border.at(n - 2) && border.at(1) != border.at(n - 1)));

	//colour copy image
	for (Point2i p : border) {
		copy.at<uchar>(p.x, p.y) = 0;
	}

	//output chaincode and compute derivative
	std::cout << "Chain code: \n";
	for (int i = 1; i < code.size(); i++) {
		derivative.push_back((code.at(i) - code.at(i - 1) + 8) % 8);
		std::cout << code.at(i) << " ";
	}

	std::cout << "\nDerivative: \n";
	for (int d : derivative)
		std::cout << d << " ";

	imshow("border", copy);
	waitKey(0);
}

void reconstruct() {
	int startx, starty, n;
	std::ifstream input;
	input.open("Images/reconstruct.txt");
	input >> startx;
	input >> starty;
	input >> n;

	vector<int> code;
	vector<Point2i> border;
	Point2i start = Point2i(startx, starty);
	Point2i curr = start;
	for (int i = 0; i < n; i++) {
		int c;
		input >> c;
		code.push_back(c);
	}

	border.push_back(start);
	for (int i = 0; i < n; i++) {
		Point2i next = getPixelAtDir(code.at(i), curr);
		border.push_back(next);
		curr = next;
	}

	Mat img = imread("Images/gray_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	Mat copy = img.clone();

	for (Point p : border) {
		copy.at<uchar>(p.x, p.y) = 0;
	}
	imshow("border", copy);
	waitKey(0);
}



Mat dilationHelper(Mat img) {
	Mat copy(img.rows, img.cols, CV_8UC1);
	img.copyTo(copy);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++)
		{
			if (img.at<uchar>(i, j) == 255)
				if (img.at<uchar>(i - 1, j) == 0 ||
					img.at<uchar>(i - 1, j - 1) == 0 ||
					img.at<uchar>(i, j - 1) == 0 ||
					img.at<uchar>(i + 1, j - 1) == 0 ||
					img.at<uchar>(i + 1, j) == 0 ||
					img.at<uchar>(i + 1, j + 1) == 0 ||
					img.at<uchar>(i, j + 1) == 0 ||
					img.at<uchar>(i - 1, j + 1) == 0)

					copy.at<uchar>(i, j) = 0;

		}
	return copy;

}

void dilation(int n) {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);
		for (int i = 0; i < n; i++)
			copy = dilationHelper(copy);
		imshow("after dilation", copy);
		waitKey(0);
	}
}

Mat erosionHelper(Mat img) {
	Mat copy(img.rows, img.cols, CV_8UC1);
	img.copyTo(copy);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) 
		{
			if (img.at<uchar>(i - 1, j) == 255 ||
				img.at<uchar>(i - 1, j - 1) == 255 ||
				img.at<uchar>(i, j - 1) == 255 ||
				img.at<uchar>(i + 1, j - 1) == 255 ||
				img.at<uchar>(i + 1, j) == 255 ||
				img.at<uchar>(i + 1, j + 1) == 255 ||
				img.at<uchar>(i, j + 1) == 255 ||
				img.at<uchar>(i - 1, j + 1) == 255)

				copy.at<uchar>(i, j) = 255;
		}
	return copy;
}

void erosion(int n) {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);
		for (int i = 0; i < n; i++)
			copy = erosionHelper(copy);
		imshow("after erosion", copy);
		waitKey(0);
	}
}

Mat erodeImage(Mat img, int n) {
	Mat copy(img.rows, img.cols, CV_8UC1);
	img.copyTo(copy);
	while (n--)
		copy = erosionHelper(copy);
	return copy;
}

Mat dilateImage(Mat img, int n) {
	Mat copy(img.rows, img.cols, CV_8UC1);
	img.copyTo(copy);
	while (n--)
		copy = dilationHelper(copy);
	return copy;
}

void opening(int n)
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);
		copy = erodeImage(copy, n);
		copy = dilateImage(copy, n);
		imshow("after opening", copy);
		waitKey(0);
	}
}

void closing(int n)
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);
		copy = dilateImage(copy, n);
		copy = erodeImage(copy, n);
		imshow("border", copy);
		waitKey(0);
	}
}

void extractBoundary()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);
		copy = erodeImage(img, 1);
		for (int i = 1; i < img.rows - 1; i++)
			for (int j = 1; j < img.cols - 1; j++){
				copy.at<uchar>(i, j) = img.at<uchar>(i, j) + 255 - copy.at<uchar>(i, j);
			}
		imshow("boundary", copy);
		waitKey(0);
	}
}

bool equalImages(Mat a, Mat b) {
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			if (a.at<uchar>(i, j) != b.at<uchar>(i, j))
				return false;
	return true;
}

Mat intersection(Mat a, Mat b) {
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			if (a.at<uchar>(i, j) == 0 && b.at<uchar>(i, j) == 0)
				a.at<uchar>(i, j) = 0;
			else
				a.at<uchar>(i, j) = 255;
	return a;
}

Mat imgUnion(Mat a, Mat b) {
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			if (a.at<uchar>(i, j) == 255 && b.at<uchar>(i, j) == 255)
				a.at<uchar>(i, j) = 255;
			else
				a.at<uchar>(i, j) = 0;
	return a;
}

void fillRegion() {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		int px = img.rows / 2, py = img.cols / 2;

		Mat complement(img.rows, img.cols, CV_8UC1);
		img.copyTo(complement);
		Mat aux(img.rows, img.cols, CV_8UC1);
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				complement.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
				aux.at<uchar>(i, j) = 255;
			}
		aux.at<uchar>(px, py) = 0;


		Mat rez(img.rows, img.cols, CV_8UC1);
		while (!equalImages(rez, aux))
		{
			aux.copyTo(rez);
			aux = dilateImage(aux, 1);
			aux = intersection(aux, complement);
		}
		imshow("fill", imgUnion(rez, img));
		waitKey(0);
	}
}

int mean_deviation() {
	int mean = 0;
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);

		
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				mean += img.at<uchar>(i, j);
			}

		mean = mean / (img.rows * img.cols);
	}
	return mean;
}

int meanGivenImage(Mat img) {
	int mean = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			mean += img.at<uchar>(i, j);
		}

	mean = mean / (img.rows * img.cols);

	return mean;
}

int std_deviation(Mat img) {
	int dev = 0;
	int mean = meanGivenImage(img);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			dev += pow((img.at<uchar>(i, j) - mean), 2);
		}

	dev = dev / (img.rows * img.cols);

	return sqrt(dev);
}

int* compute_histogram() {
	int hist[256] = {};
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);

		float pdf[256] = {};
		int m = img.rows * img.cols;

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				int val = img.at<uchar>(i, j);
				hist[val] ++;
			}

		for (int i = 0; i < 256; i++) {
			pdf[i] = (float) hist[i] / m;
		}

		showHistogram("histogram", hist, 256, 200);
		//showHistogram("pdf", pdf, 256, 200);
		waitKey(0);
	}
	return hist;
}

void threshold_computation(float error) {
	Mat img = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	Mat copy;
	img.copyTo(copy);

	int hist[256] = {};
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int val = img.at<uchar>(i, j);
			hist[val] ++;
		}

	int min = hist[0], max = hist[0];
	bool foundMin = false;
	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0) {
			if (foundMin == false) {
				min = i;
				foundMin = true;
			}
			else
				max = i;
		}
	}

	printf("min: %d max: %d\n", min, max);

	float t = (min + max) / 2;
	float tprev = 0;
	float mg1 = 0, mg2 = 0;
	int n1 = 0, n2 = 0;

	do {
		for (int f = min; f < t; f++) {
			mg1 += f * hist[f];
			n1 += hist[f];
		}

		for (int f = t + 1; f < max; f++) {
			mg2 += f * hist[f];
			n2 += hist[f];
		}

		mg1 = mg1 / n1;
		mg2 = mg2 / n2;

		tprev = t;
		t = (mg1 + mg2) / 2;
	} while (abs(t - tprev) < error);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < t)
				copy.at<uchar>(i, j) = 0;
			else
				copy.at<uchar>(i, j) = 255;
		}

	printf("threshold is: %.2f", t);
	imshow("binary", copy);
	waitKey(0);
}

void histogram_stretch(int gout_min, int gout_max) {
	Mat img = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original image", img);
	Mat copy(img.rows, img.cols, CV_8UC1);
	img.copyTo(copy);

	int hist[256] = {};
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int val = img.at<uchar>(i, j);
			hist[val] ++;
		}
	int hist2[256] = {};

	int ginmin = 0, ginmax = 0;
	bool foundMin = false;
	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0) {
			if (foundMin == false) {
				ginmin = i;
				foundMin = true;
			}
			else
				ginmax = i;
		}
	}

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int gin = img.at<uchar>(i, j);
			int gout = gout_min + (gin - ginmin) * (gout_max - gout_min) / (ginmax - ginmin);
			if (gout < 0)
				gout = 0;
			if (gout > 255)
				gout = 255;
			hist2[gout] = hist[gin];
			copy.at<uchar>(i, j) = gout;
	}


	showHistogram("before", hist, 256, 200);
	showHistogram("transformed histogram", hist2, 256, 200);
	imshow("transformed image", copy);
	waitKey(0);
}

void gamma_correction(float gamma) {
	int hist[256] = {};
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		Mat copy(img.rows, img.cols, CV_8UC1);
		img.copyTo(copy);

		int hist[256] = {};
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				int val = img.at<uchar>(i, j);
				hist[val] ++;
			}

		float l = 255;
		int hist2[256] = {};
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				int gin = img.at<uchar>(i, j);
				int gout = l * pow(gin / l, gamma);

				if (gout < 0)
					gout = 0;
				if (gout > 255)
					gout = 255;
				copy.at<uchar>(i, j) = gout;
				hist2[gout] = hist[gin];
			}

		showHistogram("before", hist, 256, 200);
		showHistogram("transformed histogram", hist2, 256, 200);
		imshow("transformed image", copy);
		waitKey(0);
	}
}

void histogram_normalization() {
	int hist[256] = {};
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		Mat copy;
		img.copyTo(copy);

		int l = 255;
		int m = img.rows * img.cols;
		float cpdf[256] = {};
		int hist2[256] = {};
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				int val = img.at<uchar>(i, j);
				hist[val] ++;
			}

		for (int k = 0; k < 256; k++) {
			for (int g = 0; g < k; g++) {
				cpdf[k] += hist[g];
			}
			hist2[k] = cpdf[k];
			cpdf[k] /= m;
		}

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				int gin = img.at<uchar>(i, j);
				int gout = l * cpdf[gin];
				copy.at<uchar>(i, j) = gout;
			}

		showHistogram("histogram", hist, 256, 200);
		showHistogram("cpdf", hist2, 256, 200);
		imshow("after", copy);
		waitKey(0);
	}
}

void convolution(Mat_<float> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

	output.create(img.size());
	memcpy(output.data, img.data, img.rows * img.cols * sizeof(uchar));

	//img.copyTo(output);

	float scalingCoeff = 1;
	float additionFactor = 0;

	//TODO: decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
	// low pass if all elements >= 0
	// high pass has elements < 0
	bool lowPass = true;
	for (int i = 0; i < filter.rows; i++)
		for (int j = 0; j < filter.cols; j++) {
			if (filter.at<float>(i, j) < 0)
				//is high pass
				lowPass = false;
		}

	// compute scaling coefficient and addition factor for low pass and high pass
	// low pass: additionFactor = 0, scalingCoeff = sum of all elements
	// high pass: formula 9.20
	if (lowPass == true) {
		additionFactor = 0;
		scalingCoeff = 0;
		for (int i = 0; i < filter.rows; i++)
			for (int j = 0; j < filter.cols; j++) {
				scalingCoeff += filter.at<float>(i, j);
			}
	}
	else {
		float posSum = 0, negSum = 0;
		for (int i = 0; i < filter.rows; i++)
			for (int j = 0; j < filter.cols; j++) {
				if (filter.at<int>(i, j) > 0)
					posSum += filter.at<float>(i, j);
				else
					negSum += filter.at<float>(i, j);
			}

		scalingCoeff = 2 * max(posSum, abs(negSum));
		additionFactor = 255 / 2;
	}

	 //TODO: implement convolution operation (formula 9.2)
	// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]
	int k = (filter.rows - 1)/ 2;
	for(int i = k; i < img.rows - k; i ++)
		for (int j = k; j < img.cols - k; j++) {
			int val = 0;
				
			for (int u = 0; u < filter.rows; u++)
				for (int v = 0; v < filter.cols; v++)
					val += filter.at<float>(u, v) * img.at<uchar>(i + u - k, j + v - k);
	
			output.at<uchar>(i, j) = val / scalingCoeff + additionFactor;
		}
}

void convolution2(Mat_<float> &filter, Mat_<uchar> &img, Mat_<int> &output) {

		output.create(img.size());
		memcpy(output.data, img.data, img.rows * img.cols * sizeof(uchar));

		//img.copyTo(output);

		float scalingCoeff = 1;
		float additionFactor = 0;

		//TODO: decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
		// low pass if all elements >= 0
		// high pass has elements < 0
		bool lowPass = true;
		for (int i = 0; i < filter.rows; i++)
			for (int j = 0; j < filter.cols; j++) {
				if (filter.at<float>(i, j) < 0)
					//is high pass
					lowPass = false;
			}

		// compute scaling coefficient and addition factor for low pass and high pass
		// low pass: additionFactor = 0, scalingCoeff = sum of all elements
		// high pass: formula 9.20
		if (lowPass == true) {
			additionFactor = 0;
			scalingCoeff = 0;
			for (int i = 0; i < filter.rows; i++)
				for (int j = 0; j < filter.cols; j++) {
					scalingCoeff += filter.at<float>(i, j);
				}
		}
		else {
			float posSum = 0, negSum = 0;
			for (int i = 0; i < filter.rows; i++)
				for (int j = 0; j < filter.cols; j++) {
					if (filter.at<int>(i, j) > 0)
						posSum += filter.at<float>(i, j);
					else
						negSum += filter.at<float>(i, j);
				}

			scalingCoeff = 2 * max(posSum, abs(negSum));
			additionFactor = 255 / 2;
		}

		// TODO: implement convolution operation (formula 9.2)
		// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]
		int k = (filter.rows - 1) / 2;
		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++) {
				float val = 0;

				for (int u = 0; u < filter.rows; u++)
					for (int v = 0; v < filter.cols; v++)
						val += filter.at<float>(u, v) * img.at<uchar>(i + u - k, j + v - k);

				output.at<int>(i, j) = val;
			}
	}

void median_filter(int w) {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);

		Mat copy;
		img.copyTo(copy);

		int k = w / 2;
		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++) {
				vector<uchar> intensities;
				for(int u = 0; u < w; u++)
					for(int v = 0; v < w; v++)
						intensities.push_back(img.at<uchar>(i + u - k, j + v - k));

				//sort vector
				std::sort(intensities.begin(), intensities.end());

				int middle = intensities.size() / 2;
				copy.at<uchar>(i, j) = intensities.at(middle);
			}
		
		imshow("filtered", copy);
		waitKey(0);
	}
}

void gaussian_filter(int w) {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);

		Mat_<uchar> copy;
		img.copyTo(copy);

		float std_dev = w / 6.0;
		//int k = w / 2;

		Mat_<float> filter(w, w, CV_32FC1);
		int x0 = (w - 1) / 2;
		int y0 = (w - 1) / 2;
		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++) {
				filter.at<float>(i, j) = (1 / (2 * PI * std_dev * std_dev)) * exp(-1 * (pow(i - x0, 2) + pow(j - y0, 2)) / (2 * std_dev * std_dev));
			}

		convolution(filter, img, copy);
		imshow("gaussian filter", copy);
		waitKey(0);
	}
}

void gaussian_filter_separated(int w) {
		char fname[MAX_PATH];
		if (openFileDlg(fname))
		{
			Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			imshow("original image", img);

			Mat_<uchar> copy;
			img.copyTo(copy);

			Mat_<uchar> aux;
			img.copyTo(aux);

			float std_dev = w / 6.0;
			//int k = w / 2;

			Mat_<float> gx(w, w, CV_32FC1);
			Mat_<float> gy(w, w, CV_32FC1);

			for (int u = 0; u < w; u++)
				for (int v = 0; v < w; v++) {
					gx(u, v) = 0;
					gy(u, v) = 0;
				}

			int x0 = (w - 1) / 2;
			int y0 = (w - 1) / 2;

			for (int i = 0; i < w; i++)
				for (int j = 0; j < w; j++) {
					gx(i, j) = (1 / (2 * PI * std_dev * std_dev)) * exp((-1 * pow(i - x0, 2)) / (2 * std_dev * std_dev));
					gy(i, j) = (1 / (2 * PI * std_dev * std_dev)) * exp((-1 * pow(j - y0, 2)) / (2 * std_dev * std_dev));
				}

			convolution(gx, img, copy);
			convolution(gy, copy, aux);
			imshow("gaussian filter", aux);
			waitKey(0);
		}
}

vector<Mat_<int>> sobel_gradient(Mat_<uchar> img) {
	Mat_<int> gradx, grady;

	Mat_<float> sobelX = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat_<float> sobelY = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

	convolution2(sobelX, img, gradx);
	convolution2(sobelY, img, grady);

	vector<Mat_<int>> res;
	res.push_back(gradx);
	res.push_back(grady);
	return res;
}

void compute_gradient() {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> gradx, grady;

		Mat_<float> sobelX = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
		Mat_<float> sobelY = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

		convolution(sobelX, img, gradx);
		convolution(sobelY, img, grady);


		imshow("grad x", gradx);
		imshow("grad y", grady);
		waitKey(0);
	}
}

int get_quantization(double angle) {
	if (angle > 15 * PI / 8 || angle < PI / 8 || (angle > 7 * PI / 8 && angle < 9 * PI / 8))
		//case 2
		return 2;
	else if ((angle > PI / 8 && angle < 3 * PI / 8) || (angle > 9 * PI / 8 && angle < 11 * PI / 8))
		return 1;
	else if ((angle > 3 * PI / 8 && angle < 5 * PI / 8) || (angle > 11 * PI / 8 && angle < 13 * PI / 8))
		return 0;
	else
		return 3;
}

Mat edges() {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> orientation = Mat(img.rows, img.cols, CV_8UC1);
		Mat_<uchar> magnitude = Mat(img.rows, img.cols, CV_8UC1);
		Mat_<uchar> dest = Mat(img.rows, img.cols, CV_8UC1);

		vector<Mat_<int>> res = sobel_gradient(img);
		Mat_<int> gradx = res[0];
		Mat_<int> grady = res[1];
		// compute magnitude
		for(int i = 0; i< img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				int gx = gradx.at<int>(i, j);
				int gy = grady.at<int>(i, j);
				magnitude.at<uchar>(i, j) = sqrt(gx * gx + gy * gy) / (4*sqrt(2));
				double angle = atan2(gy, gx);
				if (angle < 0)
					angle += 2 * PI;
				//quantization
				orientation.at<uchar>(i, j) = get_quantization(angle);
				//std::cout << angle << " ";
			}

		imshow("magnitude", magnitude);

		//gradient suppression
		for (int r = 1; r < img.rows - 1; r++)
			for (int c = 1; c < img.cols - 1; c++)
			{
				dest[r][c] = 0;
				switch (orientation[r][c]) {
				case 0:
					if (magnitude[r][c] > magnitude[r+1][c] && magnitude[r][c] > magnitude[r-1][c])
						dest[r][c] = magnitude[r][c];
					break;
				case 1:
					if (magnitude[r][c] > magnitude[r - 1][c + 1] && magnitude[r][c] > magnitude[r + 1][c - 1])
						dest[r][c] = magnitude[r][c];
					break;
				case 2:
					if (magnitude[r][c] > magnitude[r][c-1] && magnitude[r][c] > magnitude[r][c+1])
						dest[r][c] = magnitude[r][c];
					break;
				case 3:
					if (magnitude[r][c] > magnitude[r + 1][c + 1] && magnitude[r][c] > magnitude[r - 1][c - 1])
						dest[r][c] = magnitude[r][c];
					break;
				}
			}
		imshow("edges", dest);
		waitKey(0);
		return dest;
	}
}

void adaptive_thresholding(int percent) {
	Mat img = edges();
	int hist[256] = {};
	int m = img.rows * img.cols;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int val = img.at<uchar>(i, j);
			hist[val] ++;
		}

	int sum = 0;

}

int main()
{
	int op, n;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Resize image\n");
		printf(" 4 - Process video\n");
		printf(" 5 - Snap frame from live video\n");
		printf(" 6 - Mouse callback demo\n");
		printf(" 7 - L1 Negative Image \n");
		printf(" 8 - Additive factor grayscale Image \n");
		printf(" 9 - Multiplicative factor grayscale Image \n");
		printf(" 10 - Squares\n");
		printf(" 11 - Horizontal flip\n");
		printf(" 12 - Vertical flip\n");
		printf(" 13 - Crop image\n");
		printf(" 14 - Resize image\n");
		printf(" 15 - RGB to three grayscale\n");
		printf(" 16 - RGB to grayscale\n");
		printf(" 17 - Grayscale to binary\n");
		printf(" 18 - RGB to HSV\n");
		printf(" 19 - Detect stop\n");
		printf(" 20 - Detect RGB\n");
		printf(" 21 - Display object info\n");
		printf(" 22 - Label objects using BFS\n");
		printf(" 23 - Label objects using 2 pass\n");
		printf(" 24 - Trace border\n");
		printf(" 25 - Compute chain code\n");
		printf(" 26 - Reconstruct\n");
		printf(" 27 - Dilate\n");
		printf(" 28 - Erode\n");
		printf(" 29 - Opening\n");
		printf(" 30 - Closing\n");
		printf(" 31 - Boundary extraction\n");
		printf(" 32 - Region filling\n");
		printf(" 33 - Compute mean and standard deviation of image\n");
		printf(" 34 - Compute histogram\n");
		printf(" 35 - Automatic threshold computation\n");
		printf(" 36 - Histogram stretch and shrink\n");
		printf(" 37 - Histogram gamma correction\n");
		printf(" 38 - Histogram equalization\n");
		printf(" 39 - Mean filter\n");
		printf(" 40 - 2D gaussian filter\n");
		printf(" 41 - Separated gaussian filter\n");
		printf(" 42 - Compute gradient\n");
		printf(" 43 - Edges\n");
		printf(" 44 - Edges with thresholding\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testResize();
				break;
			case 4:
				testVideoSequence();
				break;
			case 5:
				testSnap();
				break;
			case 6:
				testMouseClick();
				break;
			case 7:
				negative_image();
				break;
			case 8:
				change_gray_additive();
				break;
			case 9:
				change_gray_multiplicative();
				break;
			case 10:
				create_color_image();
				break;
			case 11:
				horizontal_flip();
				break;
			case 12:
				vertical_flip();
				break;
			case 13:
				center_crop();
				break;
			case 14:
				resize();
				break;
			case 15:
				rgb_to_three();
				break;
			case 16:
				rgb_to_grayscale();
				break;
			case 17:
				grayscale_to_binary();
				break;
			case 18:
				rgb_to_hsv();
				break;
			case 19:
				detect_traffic_sign();
				break;
			case 21:
				select_object();
				break;
			case 22:
				label_BFS();
				break;
			case 23:
				two_pass();
				break;
			case 24:
				trace_border();
				break;
			case 25:
				chain_code();
				break;
			case 26:
				reconstruct();
				break;
			case 27:
				std::cin >> n;
				dilation(n);
				break;
			case 28:
				std::cin >> n;
				erosion(n);
				break;
			case 29:
				std::cin >> n;
				opening(n);
				break;
			case 30:
				std::cin >> n;
				closing(n);
				break;
			case 31:
				extractBoundary();
				break;
			case 32:
				fillRegion();
				break;
			case 33:
				char fname[MAX_PATH];
				if (openFileDlg(fname))
				{
					Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
					imshow("original image", img);

					printf("Mean deviation of image: %d\n", meanGivenImage(img));
					printf("Standard deviation: %d\n", std_deviation(img));

					waitKey(0);
				}
				break;
			case 34:
				//showHistogram("histogram", compute_histogram(), 256, 200);
				//waitKey(0);
				compute_histogram();
				break;
			case 35:
				threshold_computation(0.1);
				break;
			case 36:
				int goutmin, goutmax;
				std::cout << "Enter gin min and gin max\n";
				std::cin >> goutmin >> goutmax;
				histogram_stretch(goutmin, goutmax);
				break;
			case 37:
				float gamma;
				std::cout << "Enter gamma\n";
				std::cin >> gamma;
				gamma_correction(gamma);
				break;
			case 38:
				histogram_normalization();
				break;
			case 39:
				std::cout << "Enter dimension\n";
				int w;
				std::cin >> w;
				median_filter(w);
				break;
			case 40:
				std::cout << "Enter dimension\n";
				int dim;
				std::cin >> dim;
				gaussian_filter(dim);
				break;
			case 41:
				std::cout << "Enter dimension\n";
				int x;
				std::cin >> x;
				gaussian_filter_separated(x);
				break;
			case 42:
				compute_gradient();
				break;
			case 43:
				edges();
				break;
		}
	}
	while (op!=0);
	return 0;
}