// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <random>
#define hist_size 256*3
#define e 2.71828


struct peak {
	int teta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
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
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
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
		if (c == 115) { //'s' pressed - snapp the image to a file
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

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
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

void lab1_afisare_puncte(char* filename)
{

	FILE* f = fopen(filename, "r");
	int n;
	fscanf(f, "%d", &n);
	float** puncte = (float**)calloc(n, sizeof(float*));
	for (int i = 0; i < n; i++)
	{
		puncte[i] = (float*)calloc(2, sizeof(float));
	}
	for (int i = 0; i < n; i++)
	{
		fscanf(f, "%f %f", &puncte[i][1], &puncte[i][0]);
	}

	Mat img(500, 500, CV_8UC3); //8bit unsigned 3 channel
	img.setTo({ 255 });

	for (int i = 0; i < n; i++)
	{
		if (puncte[i][0] > 0 && puncte[i][1] > 0) {
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[0] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[1] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[2] = 0;
		}
	}



	imshow("puncte", img);
	waitKey();
	fclose(f);
}

void lab1_model1(char* filename)
{
	FILE* f = fopen(filename, "r");
	int n;
	fscanf(f, "%d", &n);
	float** puncte = (float**)calloc(n, sizeof(float*));
	for (int i = 0; i < n; i++)
	{
		puncte[i] = (float*)calloc(2, sizeof(float));
	}
	for (int i = 0; i < n; i++)
	{
		fscanf(f, "%f %f", &puncte[i][1], &puncte[i][0]);
	}

	Mat img(500, 500, CV_8UC3); //8bit unsigned 3 channel
	img.setTo({ 255 });

	for (int i = 0; i < n; i++)
	{
		if (puncte[i][0] > 0 && puncte[i][1] > 0) {
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[0] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[1] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[2] = 0;
		}
	}

	double teta0, teta1;
	float sumaProd = 0, sumax = 0, sumaPatratX = 0, sumay = 0;
	for (int i = 0; i < n; i++)
	{
		sumaProd += puncte[i][0] * puncte[i][1];
		sumax += puncte[i][1];
		sumay += puncte[i][0];
		sumaPatratX += puncte[i][1] * puncte[i][1];
	}
	teta1 = (n * sumaProd - sumax * sumay) / (n * sumaPatratX - sumax * sumax);
	teta0 = 1.0 / n * (sumay - teta1 * sumax);

	line(img, Point(1, teta0 + teta1 * 1), Point(499, teta0 + teta1 * 499), Scalar(0, 0, 255));

	imshow("linie met1", img);
	waitKey();
}

void lab1_model2(char* filename)
{
	FILE* f = fopen(filename, "r");
	int n;
	fscanf(f, "%d", &n);
	float** puncte = (float**)calloc(n, sizeof(float*));
	for (int i = 0; i < n; i++)
	{
		puncte[i] = (float*)calloc(2, sizeof(float));
	}
	for (int i = 0; i < n; i++)
	{
		fscanf(f, "%f %f", &puncte[i][1], &puncte[i][0]);
	}

	Mat img(500, 500, CV_8UC3); //8bit unsigned 3 channel
	img.setTo({ 255 });

	for (int i = 0; i < n; i++)
	{
		if (puncte[i][0] > 0 && puncte[i][1] > 0) {
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[0] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[1] = 0;
			img.at<Vec3b>(puncte[i][0], puncte[i][1])[2] = 0;
		}
	}

	float beta, alpha;
	float sumaProd = 0, sumax = 0, difPatrate = 0, sumay = 0;
	for (int i = 0; i < n; i++)
	{
		sumaProd += puncte[i][0] * puncte[i][1];
		sumax += puncte[i][1];
		sumay += puncte[i][0];
		difPatrate += puncte[i][0] * puncte[i][0] - puncte[i][1] * puncte[i][1];
	}
	beta = -1.0 / 2 * atan2(2 * sumaProd - 2.0 / n * sumax * sumay, difPatrate + 1.0 / n * sumax * sumax - 1.0 / n * sumay * sumay);
	alpha = 1.0 / n * (cos(beta) * sumax + sin(beta) * sumay);

	if (abs(sin(beta)) > 0.1) {
		line(img, Point(1, (alpha - cos(beta)) / sin(beta)), Point(499, (alpha - 499 * cos(beta)) / sin(beta)), Scalar(0, 0, 255));
	}
	else
	{
		line(img, Point((alpha - sin(beta)) / cos(beta), 1), Point((alpha - 499 * sin(beta)) / cos(beta), 499), Scalar(0, 0, 255));

	}
	imshow("linie met2", img);
	waitKey();
}

void lab2_ransac(char* path)
{
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	int m = img.rows;
	int n = img.cols;

	std::vector<Point> v;

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				v.push_back({ j, i });
			}
		}
	}

	printf("\nNumar de puncte negre = %d\n", v.size());

	float t = 10.0, p = 0.99, q = 0.8, s = 2.0;

	if (strcmp(path, "Imglab2/points1.bmp") == 0)
	{
		q = 0.3;
	}
	int nr = v.size();
	float N = log(1 - p) / log(1 - pow(q, s));
	float T = q * nr;

	printf("q = %f\nN = %f\nT = %f\n", q, N, T);
	srand(time(NULL));
	int maxct = 0;
	int maxa = 0, maxb = 0, maxc = 0;
	for (int i = 0; i < floor(N); i++)
	{
		int i1 = rand() % nr;
		int i2 = rand() % nr;
		while (i1 == i2)
		{
			i2 = rand() % nr;
		}
		printf("%d %d\n", i1, i2);
		int a = v[i1].y - v[i2].y;
		int b = v[i2].x - v[i1].x;
		int c = v[i1].x * v[i2].y - v[i2].x * v[i1].y;

		int ct = 0;
		for (int j = 0; j < nr; j++)
		{
			float dist = abs(a * v[j].x + b * v[j].y + c) * 1.0 / sqrt(a * a + b * b);
			if (dist <= t)
			{
				ct++;
			}

		}

		if (maxct < ct)
		{
			maxct = ct;
			maxa = a;
			maxb = b;
			maxc = c;
		}

		if (ct > T)
		{
			break;
		}


	}

	if (abs(maxb) > 5)
	{
		line(img, Point(1, (-maxc - maxa * 1) / maxb), Point(n - 1, (-maxc - maxa * (n - 1)) / maxb), Scalar(0, 0, 0));
	}
	else
	{
		line(img, Point((-maxc - maxb * 1) / maxa, 1), Point((-maxc - maxb * (m - 1)) / maxa, m - 1), Scalar(0, 0, 0));
	}

	imshow("img", img);
	waitKey();
}

void lab3_hough(char* path)
{
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	int m = img.rows;
	int n = img.cols;
	int D = sqrt(m * m + n * n);


	Mat Hough(D + 1, 360, CV_32SC1);
	Hough.setTo(0);

	float roMax = 0;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (img.at<uchar>(i, j) == 255) {
				for (int teta = 0; teta < 360; teta++) {
					float tetaRad = teta * PI / 180;
					float ro = j * cos(tetaRad) + i * sin(tetaRad);
					if (ro > 0) {
						Hough.at<int>(ro, teta)++;
					}
					if (ro > roMax) {
						roMax = ro;
					}
				}
			}
		}
	}

	int maxHough = 0;
	for (int i = 0; i < D + 1; i++) {
		for (int j = 0; j < 360; j++) {
			if (Hough.at<int>(i, j) > maxHough) {
				maxHough = Hough.at<int>(i, j);
			}
		}
	}

	Mat HoughImg;
	Hough.convertTo(HoughImg, CV_8UC1, 255.f / maxHough);

	imshow("Acumulator", HoughImg);
	waitKey();

	std::vector<peak> v;
	for (int ro = 0; ro < D + 1; ro++) {
		for (int teta = 0; teta < 360; teta++) {
			int ok = 1;
			for (int dro = -3; dro < 4; dro++) {
				for (int dteta = -3; dteta < 4; dteta++) {
					if (ro + dro >= 0 && ro + dro < D + 1 && teta + dteta >= 0 && teta + dteta < 360) {
						if (Hough.at<int>(ro + dro, teta + dteta) > Hough.at<int>(ro, teta)) {
							ok = 0;
						}
					}
				}
			}
			if (ok == 1) {
				v.push_back({ teta, ro, Hough.at<int>(ro, teta) });
			}
		}
	}
	printf("\n%d\n", v.size());
	//std::vector<peak> sortedv;
	sort(v.begin(), v.end());
	int k = 9;
	for (int i = 0; i < k; i++)
	{
		int teta = v[i].teta;
		int ro = v[i].ro;
		printf("ro: %d teta: %d\n", ro, teta);
		float tetaRad = teta * PI / 180;
		if (abs(sin(tetaRad)) > 0.1) {

			line(img, Point(1, (ro - cos(tetaRad)) / sin(tetaRad)), Point(n - 1, (ro - (n - 1) * cos(tetaRad)) / sin(tetaRad)), Scalar(255, 255, 255));
		}
		else
		{
			line(img, Point((ro - sin(tetaRad)) / cos(tetaRad), 1), Point((ro - (m - 1) * sin(tetaRad)) / cos(tetaRad), m - 1), Scalar(255, 255, 255));

		}
	}

	imshow("linii1", img);
	waitKey();
}

Mat calc_DT(Mat img, int m, int n) {
	Mat dt = img.clone();

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (img.at<uchar>(i, j) > 0) {
				dt.at<uchar>(i, j) = 255;
			}
		}
	}

	int di[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int dj[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int weight[] = { 3, 2, 3, 2, 0, 2, 3, 2, 3 };

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int min = INT_MAX;
			for (int k = 0; k < 9; k++) {
				if (i + di[k] >= 0 && i + di[k] < m && j + dj[k] >= 0 && j + dj[k] < n) {
					int res = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (min > res) {
						min = res;
					}
				}
			}
			dt.at<uchar>(i, j) = min;
		}
	}

	for (int i = m - 1; i >= 0; i--) {
		for (int j = n - 1; j >= 0; j--) {
			int min = INT_MAX;
			for (int k = 0; k < 9; k++) {
				if (i + di[k] >= 0 && i + di[k] < m && j + dj[k] >= 0 && j + dj[k] < n) {
					int res = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (min > res) {
						min = res;
					}
				}
			}
			dt.at<uchar>(i, j) = min;
		}
	}

	return dt;
}

void lab4_DT(char* path) {
	Mat img = imread(path, IMREAD_GRAYSCALE);
	int m = img.rows;
	int n = img.cols;

	Mat dt = calc_DT(img, m, n);

	imshow("Transformata distanta", dt);
	waitKey();
}

float calc_scor(Mat dtTemp, Mat imgUnk, int m, int n, Point cTemp, Point cUnk, int centered) {
	std::vector<Point> contour;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (imgUnk.at<uchar>(i, j) == 0) {
				contour.push_back({ i, j });
			}
		}
	}

	int sum = 0;
	for (int i = 0; i < contour.size(); i++) {
		if (centered == 1) {
			int p = contour[i].x + cTemp.x - cUnk.x;
			int q = contour[i].y + cTemp.y - cUnk.y;
			if (p < dtTemp.rows && p >= 0 && q < dtTemp.cols && q >= 0) {
				sum += dtTemp.at<uchar>(contour[i].x + cTemp.x - cUnk.x, contour[i].y + cTemp.y - cUnk.y);
			}
		}
		else {
			sum += dtTemp.at<uchar>(contour[i].x, contour[i].y);

		}
	}

	return sum * 1.0 / contour.size();
}

Point calc_centru_de_masa(Mat img, int m, int n) {
	int sumx = 0;
	int sumy = 0;
	int sumcont = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (img.at<uchar>(i, j) == 0) {
				sumx += i;
				sumy += j;
				sumcont++;
			}
		}
	}
	return { sumx / sumcont, sumy / sumcont };
}

void lab4_pattern_matching(char* fTemp, char* fUnk) {
	Mat imgTemp = imread(fTemp, IMREAD_GRAYSCALE);
	Mat imgUnk = imread(fUnk, IMREAD_GRAYSCALE);

	int m1 = imgTemp.rows;
	int n1 = imgTemp.cols;

	int m2 = imgUnk.rows;
	int n2 = imgUnk.cols;

	Mat dtTemp = calc_DT(imgTemp, m1, n1);

	printf("Score 1 for unknown without center: %f\n", calc_scor(dtTemp, imgUnk, m2, n2, { 0,0 }, { 0,0 }, 0));

	Mat dtUnk = calc_DT(imgUnk, m2, n2);
	printf("Score 2 for unknown without center: %f\n", calc_scor(dtUnk, imgTemp, m1, n1, { 0,0 }, { 0,0 }, 0));



	int cxTemp, cyTemp, cxUnk, cyUnk;

	Point cTemp = calc_centru_de_masa(imgTemp, m1, n1);
	Point cUnk = calc_centru_de_masa(imgUnk, m2, n2);

	printf("Score for unknown centered %f\n", calc_scor(dtTemp, imgUnk, m2, n2, cTemp, cUnk, 1));
	int d;
	std::cin >> d;
}

/*
	adaugam intr-un vector punctele de contur din imaginea necunoscuta
	aplicam dt pe template
	facem media valorilor din punctele care corespund in dt din template cu valorile din vector
*/

void lab5_statistics() {
	char folder[256] = "faces";
	char fname[256];
	int p = 400;
	int N = 19 * 19;
	Mat I = Mat(p, N, CV_8UC1);
	int k = 0;
	for (int i = 1; i <= 400; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, 0);

		for (int j = 0; j < N; j++) {
			I.at<uchar>(k, j) = img.at<uchar>(j / 19, j % 19);
		}
		k++;
	}

	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", I.at<uchar>(i, j));
		}
		printf("\n");
	}*/

	float med[361];

	for (int i = 0; i < N; i++) {
		int sum = 0;
		for (int j = 0; j < p; j++) {
			sum += I.at<uchar>(j, i);
		}
		med[i] = sum * 1.0 / p;
	}

	FILE* f = fopen("csv/med.csv", "w");

	for (int i = 0; i < N; i++) {
		if (i == N - 1) {
			fprintf(f, "%f", med[i]);
		}
		else {
			fprintf(f, "%f,", med[i]);
		}
	}

	fclose(f);


	Mat cov = Mat(p, N, CV_32FC1);
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < N; j++) {
			float prod = 0;
			for (int k = 1; k < p; k++) {
				prod += (I.at<uchar>(k, i) - med[i]) * (I.at<uchar>(k, j) - med[j]);
			}

			cov.at<float>(i, j) = prod * 1.0 / p;
		}

	}

	printf("%f\n", cov.at<float>(0, 0));

	//for(int i = 0; i <)
	int d;
	scanf("%d", &d);
}

void lab6_components_analizing(char* path) {
	FILE* f = fopen(path, "r");

	int n = 0, d = 0;

	fscanf(f, "%d %d", &n, &d);

	Mat X(n, d, CV_64FC1);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			fscanf(f, "%lf", &X.at<double>(i, j));
		}
	}

	double* med = (double*)calloc(d, sizeof(double));

	for (int i = 0; i < d; i++) {
		double sum = 0;
		for (int j = 0; j < n; j++) {
			sum += X.at<double>(j, i);
		}
		med[i] = sum / n;
	}

	Mat X2(n, d, CV_64FC1);

	for (int i = 0; i < d; i++) {
		for (int j = 0; j < n; j++) {
			X2.at<double>(j, i) = X.at<double>(j, i) - med[i];
		}
	}

	Mat C = X2.t() * X2 / (n - 1);

	Mat Lambda, Q;

	eigen(C, Lambda, Q);
	Q = Q.t();
	printf("Prima valoare proprie:%f\n", Lambda.at<double>(0,0));

	int g;
	scanf("%d", &g);

	Mat Xcoef = X * Q;

	int k = 1;
	Mat Xaprox(n, d, CV_64FC1);
	Xaprox.setTo(0);
	for (int i = 0; i < k; i++) {
		Xaprox += X * Q.col(i) * Q.col(i).t();
	}
	
	double sum1 = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			sum1 += fabs(X.at<double>(i, j) - Xaprox.at<double>(i, j));
		}
	}

	double medie = sum1 / (n * d);


	printf("Medie: %f\n", medie);

	scanf("%d", &g);

	int b = 0;
	if (strcmp(path, "DataLab6/pca2d.txt") == 0) {
		b = 2;
		
	}
	else {
		b = 3;
	}

	double min1 = Xcoef.at<double>(0, 0);
	double max1 = Xcoef.at<double>(0, 0); 
	double min2 = Xcoef.at<double>(0, 1);
	double max2 = Xcoef.at<double>(0, 1);
	double min3;
	double max3;
	if (strcmp(path, "DataLab6/pca3d.txt") == 0) {
		min3 = Xcoef.at<double>(0, 2);
		max3 = Xcoef.at<double>(0, 2);
	}
	for (int i = 0; i < n; i++) {
		if (min1 > Xcoef.at<double>(i, 0)) {
			min1 = Xcoef.at<double>(i, 0);
		}
		if (min2 > Xcoef.at<double>(i, 1)) {
			min2 = Xcoef.at<double>(i, 1);
		}
		if (max1 < Xcoef.at<double>(i, 0)) {
			max1 = Xcoef.at<double>(i, 0);
		}
		if (max2 < Xcoef.at<double>(i, 1)) {
			max2 = Xcoef.at<double>(i, 1);
		}
		if (strcmp(path, "DataLab6/pca3d.txt") == 0) {
			if (min3 > Xcoef.at<double>(i, 2)) {
				min3 = Xcoef.at<double>(i, 2);
			}
			if (max3 < Xcoef.at<double>(i, 2)) {
				max3 = Xcoef.at<double>(i, 2);
			}
		}
	}

	
	Mat afis(max1 - min1 + 1, max2 - min2 + 1, CV_8UC1);
	afis.setTo(255);
	for (int i = 0; i < n; i++) {
		if (strcmp(path, "DataLab6/pca2d.txt") == 0) {
			afis.at<uchar>(Xcoef.at<double>(i, 0) - min1, Xcoef.at<double>(i, 1) - min2) = 0;
		}
		else {
			afis.at<uchar>(Xcoef.at<double>(i, 0) - min1, Xcoef.at<double>(i, 1) - min2) = 
				255 - (Xcoef.at<double>(i, 2) - min3) / (max3 - min3) * 255;
		}
		
	}
	imshow("coef", afis);
	waitKey();
} 

std::pair<Mat, Mat> kmeans(Mat x, int K, int dim) {
	int n = x.rows; // numarul de puncte
	int d = x.cols; // dimensiunea punctelor
	Mat C(K, d, CV_32SC1); // centre 
	Mat L(n, 1, CV_32SC1);
	L.setTo(0);
	std::default_random_engine gen;
	std::uniform_int_distribution<int> distribution(0, n - 1);
	int* rand = (int*)malloc(K * sizeof(int));
	for (int i = 0; i < K; i++) {
		rand[i] = distribution(gen);
		for (int j = 0; j < i; j++) {
			if (rand[j] == rand[i]) {
				rand[i] = distribution(gen);
			}
		}
		for (int p = 0; p < dim; p++) {
			C.at<int>(i, p) = x.at<int>(rand[i], p);
		}
	}
	std::cout << C;
	int changed = 1;
	int iter = 0;
	int maxIter = 20;
	while (changed && iter < maxIter) {
		changed = 0;
		for (int i = 0; i < n; i++) {
			float* d = (float*)malloc(K * sizeof(float));
			for (int k = 0; k < K; k++) {
				int sum = 0;
				for (int q = 0; q < dim; q++) {
					sum += pow(x.at<int>(i, q) - C.at<int>(k, q), 2);
				}
				d[k] = sqrt(sum);
			}
			float dmin = d[0];
			int celmaiaproape = 0;
			for (int k = 0; k < K; k++) {
				if (dmin > d[k]) {
					dmin = d[k];
					celmaiaproape = k;
				}
			}
			if (L.at<int>(i, 0) != celmaiaproape) {
				changed = 1;
			}
			else {
				changed = 0;
			}

			L.at<int>(i, 0) = celmaiaproape;
			
		}
		std::cout << L;
		for (int k = 0; k < K; k++) {
			for (int j = 0; j < dim; j++) {
				int s = 0;
				int ct = 0;
				for (int i = 0; i < n; i++) {
					if (L.at<int>(i, 0) == k) {
						s += x.at<int>(i, j);
						ct++;
					}
				}
				
				C.at<int>(k, j) = s / ct;
				
				
			}
		}
	}
	return { C, L };
}

void lab7_k_means_2D(char* path) {
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	int m = img.rows;
	int n = img.cols;

	std::vector<Point> points;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (img.at<uchar>(i, j) < 255) {
				points.push_back({ i, j });
			}
		}
	}

	Mat x(points.size(), 2, CV_32SC1);
	for (int i = 0; i < points.size(); i++) {
		x.at<int>(i, 0) = points[i].x;
		x.at<int>(i, 1) = points[i].y;
	}

	std::pair<Mat, Mat> res = kmeans(x, 7, 2);

	std::default_random_engine gen;
	std::uniform_int_distribution<int> distribution(0, 255);


	const int K = 7;
	Vec3b colors[K];
	for (int i = 0; i < K; i++) {
		colors[i] = { (uchar)distribution(gen),
		(uchar)distribution(gen),
		(uchar)distribution(gen) };
	}
	
	Mat imgres(m, n, CV_8UC3);
	imgres.setTo(255);
	for (int k = 0; k < K; k++) {
		imgres.at<Vec3b>(res.first.at<int>(k, 0), res.first.at<int>(k, 1)) = colors[k];
		circle(imgres, Point(res.first.at<int>(k, 1), res.first.at<int>(k, 0)), 10, colors[k]);
	}
	
	int p = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (img.at<uchar>(i, j) < 255) {
				imgres.at<Vec3b>(i, j) = colors[res.second.at<int>(p++, 0)];
			}
		}
		
	}

	imshow("res", imgres);
	waitKey();
}

void calcHist(Mat img, float* hist) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img.at<Vec3b>(i, j)[0]]++;
			hist[img.at<Vec3b>(i, j)[1] + 256]++;
			hist[img.at<Vec3b>(i, j)[2] + 256 * 2]++;
		}
	}
	for (int i = 0; i < hist_size; i++) {
		hist[i] /= img.rows * img.cols;
	}
}

int classify(Mat img, Mat X, Mat Y) {
	float* hist = (float*)calloc(256 * 3, sizeof(float));

	calcHist(img, hist);
	std::vector<std::pair<double, int>> v;
	
	for (int i = 0; i < X.rows; i++) {
		float sum = 0;
		for (int j = 0; j < hist_size; j++) {
			sum += pow(hist[j] - X.at<float>(i, j), 2);
		}
		double dist = sqrt(sum);
		v.push_back({ dist, i });
	}
	std::sort(v.begin(), v.end());

	int* voturi = (int*)calloc(6, sizeof(int));
	int k = 30;
	for (int i = 0; i < k; i++) {
		voturi[Y.at<uchar>(v[i].second, 0)]++;
	}
	int max = 0;
	int imax = 0;
	for (int i = 0; i < 6; i++) {
		if (max < voturi[i]) {
			max = voturi[i];
			imax = i;
		}
	}
	return imax;
}

void lab8_k_nearest_neighbors() {
	Mat X(672, 256*3, CV_32FC1);
	Mat Y(672, 1, CV_8UC1);
	const int nrclasses = 6;
	char classes[nrclasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };
	int c = 0, fileNr = 0, rowX = 0;
	char fname[40];
	while (1) {
		sprintf(fname, "knn_img/train/%s/%06d.jpeg", classes[c], fileNr++);
		float* hist = (float*)calloc(256 * 3, sizeof(float));
		Mat img = imread(fname);
		if (img.cols == 0) {
			c++;
			if (c >= 6) {
				break;
			}
			fileNr = 0;
			continue;
		}
		calcHist(img, hist);
		for (int d = 0; d < hist_size; d++)
			X.at<float>(rowX, d) = hist[d];
		Y.at<uchar>(rowX, 0) = c;
		rowX++;
		
		strcpy(fname, "");
	}
	int u;
	std::cout << rowX << '\n';
	std::cin >> u;
	
	rowX = 0; 
	fileNr = 0;
	c = 0;
	Mat Conf(6, 6, CV_32SC1);
	Conf.setTo(0);
	while (1) {
		sprintf(fname, "knn_img/test/%s/%06d.jpeg", classes[c], fileNr++);
		float* hist = (float*)calloc(256 * 3, sizeof(float));
		Mat img = imread(fname);
		if (img.cols == 0) {
			c++;
			if (c >= 6) {
				break;
			}
			fileNr = 0;
			continue;
		}
		
		int p = classify(img, X, Y);
		if (rowX < 5) {
			std::cout << " Clasa " << p << "\n";
		}
		
		Conf.at<int>(c, p)++;

		rowX++;

		strcpy(fname, "");
	}

	std::cout << "Confusion Matrix\n";
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			std::cout << Conf.at<int>(i, j) << " ";
		}
		std::cout << '\n';
	}
	
	int sum = 0;
	for (int i = 0; i < 6; i++) {
		sum += Conf.at<int>(i, i);
	}
	std::cout << "Acuratete: " << sum * 1.0 / rowX * 100 << '\n';
	std::cin >> u;
}

int classifyBayes(Mat img, Mat priors, Mat likelihood) {
	double pmax = -1e6;
	
	int clasa = 0;

	for (int c = 0; c < priors.rows; c++) {
		double sumlikei = 0;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) > 128) {
					sumlikei += log(1 - likelihood.at<double>(c, i * img.cols + j));
				}
				else
				{
					sumlikei += log(likelihood.at<double>(c, i * img.cols + j));
				}
			}
		}
		
		double lnpi = log(priors.at<double>(c, 0)) + sumlikei;
		if (pmax < lnpi) {
			clasa = c;
			pmax = lnpi;
		}
	}
	
	return clasa;

}

void lab9_naive_bayes() {
	const int nrimg = 200;
	const int d = 28 * 28;
	Mat X(nrimg, d, CV_8UC1);
	

	const int C = 2; //number of classes
	Mat Y(C, d, CV_8UC1);
	for (int i = 0; i < C; i++) {
		for (int j = 0; j < d; j++) {
			Y.at<uchar>(i, j) = i + 1;
		}
	}
	char fname[256];
	int c = 0;
	int index = 0;
	while (c < C) {
		while (index < 100) {
			sprintf(fname, "naive_bayes_img/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) break;
			//process img
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) > 128) {
						X.at<uchar>(c * 100 + index, i * 28 + j) = 255;
					}
					else {
						X.at<uchar>(c * 100 + index, i * 28 + j) = 0;
					}
				}
			}
			index++;
		}
		index = 0;
		c++;
	}
	
	Mat img1(28, 28, CV_8UC1);
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			img1.at<uchar>(i, j) = X.at<uchar>(0, i * 28 + j);
		}
	}

	imshow("first", img1);
	waitKey();

	imshow("x", X);
	waitKey();

	Mat priors(C, 1, CV_64FC1);

	for (int i = 0; i < priors.rows; i++) {
		priors.at<double>(i, 0) = 100 / nrimg;
	}

	Mat likelihood(C, d, CV_64FC1);


	/*for (c = 0; c < Y.rows; c++) {

		for (int i = 0; i < d; i++) {
			int nr_inst_trasatura = 0;
			for (int j = 0; j < nrimg / Y.rows - 1; j++) {
				if (X.at<uchar>(j, i) == 255) {
					nr_inst_trasatura++;
				}
			}
			likelihood.at<double>(Y.at<uchar>(c, 0) - 1, i) = (nr_inst_trasatura + 1.0) / (nrimg / Y.rows + 2);
		}

	}*/

	int* inst_tras = (int*)calloc(C, sizeof(int));
	int k = 0;
	for (c = 0; c < Y.rows; c++) {
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < nrimg / Y.rows - 1; j++) {
				if (X.at<uchar>(j, i) == 255 && Y.at<uchar>(c, i) == c + 1) {
					inst_tras[c]++;
				}
			}
			likelihood.at<double>(Y.at<uchar>(c, 0) - 1, i) = (inst_tras[c] + 1.0) / (nrimg / Y.rows + 2);
		}
	}
	

	Mat imgTest = imread("naive_bayes_img/test/1/000000.png");

	printf("%d", classifyBayes(imgTest, priors, likelihood));

	int g;
	scanf("%d", &g);

}

struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	int classify(Mat X) {
		if (X.at<float>(feature_i) < threshold)
			return class_label;
		else
			return -class_label;
	}
};


#define MAXT 10000
struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	int classify(Mat X) {
		int sum = 0;
		for (int i = 0; i < T; i++) {
			sum += alphas[i] * hs[i].classify(X);
		}

		if (sum < 0) {
			return -1;
		}
		else {
			return 1;
		}
	}
};

void drawBoundary(Mat img, classifier cl) {
	Mat imgres = img.clone();

	for (int i = 0; i < imgres.rows; i++) {
		for (int j = 0; j < imgres.cols; j++) {
			if (!((img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0) ||
				(img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[2] == 0 && img.at<Vec3b>(i, j)[1] == 0))) {
				Mat vi = Mat(1, 2, CV_32SC1);
				vi.at<int>(0, 0) = i;
				vi.at<int>(0, 1) = j;

				if (cl.classify(vi) < 0) {
					imgres.at<Vec3b>(i, j) = Vec3b(150, 150, 0);
				}
				else {
					imgres.at<Vec3b>(i, j) = Vec3b(0, 150, 150);
				}
				
			}
		}
	}

	imshow("rez", imgres);
	waitKey();
}


weaklearner findWeakLearner(Mat X, Mat y, Mat w, int width, int height) {
	weaklearner best_h;

	float best_err = FLT_MAX;

	float z = 0;
	for (int j = 0; j < X.cols; j++) {
		for (int thr = 0; thr < (j==0?width:height); thr++) {
			for (int class_label = -1; class_label <= 1; class_label += 2) {
				float er = 0;
				for (int i = 0; i < X.rows; i++) {
					if (X.at<int>(i, j) < thr) {
						z = class_label;
					}
					else {
						z = -class_label;
					}
					if (z * y.at<int>(i, 0) < 0) {
						er += w.at<float>(i, 0);
					}
				}
				if (e < best_err) {
					best_err = e;
					best_h = { j, thr, class_label, e };
				}
			}

		}
	}

	return best_h;
}

void lab11_adaboost(char* path) {
	Mat src = imread(path, CV_LOAD_IMAGE_COLOR);
	int n = 0;
	imshow("sursa", src);
	waitKey();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j)[0] < 255 || src.at<Vec3b>(i, j)[1] < 255 || src.at<Vec3b>(i, j)[2] < 255) {
				n++;
			}
		}
	}

	Mat X(n, 2, CV_32SC1);
	Mat Y(n, 1, CV_32SC1);
	int k = 0;
	int o = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j)[0] < 255 || src.at<Vec3b>(i, j)[1] < 255 || src.at<Vec3b>(i, j)[2] < 255) {
				X.at<int>(k, 0) = i;
				X.at<int>(k, 1) = j;
				if (src.at<Vec3b>(i, j)[2] == 255 && src.at<Vec3b>(i, j)[0] == 0 && src.at<Vec3b>(i, j)[1] == 0) {
					Y.at<int>(k, 0) = 1;
					o++;
				}
				else {
					Y.at<int>(k, 0) = -1;
				}
				k++;
			}
		}
	}
	
	/*for (int i = 0; i < k; i++) {
		printf("%d\n", Y.at<uchar>(i, 0));
	}*/
	printf("Numar puncte rosii: %d\n", o);

	int c = 0;


	Mat W(n, 1, CV_32FC1);

	for (int i = 0; i < n; i++) {
		W.at<float>(i, 0) = 1.0 / n;
	}

	classifier adaboost;
	adaboost.T = 13;
	for (int i = 0; i < adaboost.T; i++) {
		weaklearner learner = findWeakLearner(X, Y, W, src.rows, src.cols);
		adaboost.hs[i] = learner;
		adaboost.alphas[i] = 0.5 * log((1 - learner.error) / learner.error);

		int s = 0;

		for (int j = 0; j < n; j++) {
			Mat Xi(1, 2, CV_32SC1);
			Xi.at<int>(0, 0) = X.at<int>(i, 0);
			Xi.at<int>(0, 1) = X.at<int>(i, 1);
			W.at<float>(j, 0) = W.at<float>(j, 0) * exp(-adaboost.alphas[i] * Y.at<int>(i, 0) * adaboost.hs[i].classify(Xi));
			s += W.at<float>(j, 0);
		}

		for (int j = 0; j < n; j++) {
			W.at<float>(j, 0) /= s;
		}
	}

	drawBoundary(src, adaboost);


	scanf("%d", &c);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab1 - Read points and print\n");
		printf(" 11 - Lab1 - Draw line model 1\n");
		printf(" 12 - Lab1 - Draw line model 2\n");
		printf(" 13 - Lab2 - RANSAC algorithm\n");
		printf(" 14 - Lab3 - Detectia dreptelor cu transformata Hough\n");
		printf(" 15 - Lab4 - DT\n");
		printf(" 16 - Lab4 - Model Recognition using DT\n");
		printf(" 17 - lab5 - Statistics\n");
		printf(" 18 - lab6 - More important components analizing\n");
		printf(" 19 - lab7 - K means\n");
		printf(" 20 - lab8 - K nearest neighbors\n");
		printf(" 21 - lab9 - Naive Bayes classification\n");
		printf(" 23 - lab11 - AdaBoost algorithm\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
		{
			printf("Fisier : ");
			char* filename = (char*)calloc(12, sizeof(char));
			scanf("%s", filename);
			lab1_afisare_puncte(filename);
			break;
		}
		case 11:
		{
			printf("Fisier : ");
			char* filename = (char*)calloc(12, sizeof(char));
			scanf("%s", filename);
			lab1_model1(filename);
			break;
		}
		case 12:
		{
			printf("Fisier : ");
			char* filename = (char*)calloc(12, sizeof(char));
			scanf("%s", filename);
			lab1_model2(filename);
			break;
		}
		case 13:
		{
			printf("Cale fisier: ");
			char filename[17];
			scanf("%s", filename);
			char fullpath[27];
			strcpy(fullpath, "Imglab2/");
			strcat(fullpath, filename);
			lab2_ransac(fullpath);
			break;
		}
		case 14:
		{
			char filename[] = "Imglab3/edge_complex.bmp";
			lab3_hough(filename);
			break;
		}
		case 15:
		{
			char filename[] = "images_DT_PM/DT/contour3.bmp";
			lab4_DT(filename);
			break;
		}
		case 16:
		{
			char filenameTemplate[] = "images_DT_PM/PatternMatching/template.bmp";
			char filenameUnknown[] = "images_DT_PM/PatternMatching/unknown_object1.bmp";
			lab4_pattern_matching(filenameTemplate, filenameUnknown);
			break;
		}
		case 17:
		{
			lab5_statistics();
			break;
		}
		case 18:
		{
			char path[] = "DataLab6/pca3d.txt";
			lab6_components_analizing(path);
			break;
		}
		case 19:
		{
			char path[] = "kmeans/points3.bmp";
			lab7_k_means_2D(path);
			break;
		}
		case 20:
		{
			lab8_k_nearest_neighbors();
			break;
		}
		case 21:
		{
			lab9_naive_bayes();
			break;
		}
		case 23:
		{
			char path[40];
			strcpy(path, "adaboost_img/points");
			char get[4];
			scanf("%s", get);
			strcat(path, get);
			strcat(path, ".bmp");
			puts(path);
			lab11_adaboost(path);
		}
		}
	} while (op != 0);
	return 0;
}