// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <random>


struct peak {
	int teta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

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
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
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
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
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

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

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

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
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
		Canny(grayFrame,edges,40,100,3);
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
		line(img, Point((alpha - sin(beta)) / cos(beta), 1), Point((alpha - 499 * sin(beta)) / cos(beta),  499), Scalar(0, 0, 255));

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
					if (ro+dro >= 0 && ro + dro < D + 1 && teta + dteta >= 0 && teta + dteta < 360) {
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
			
			line(img, Point(1, (ro - cos(tetaRad)) / sin(tetaRad)), Point(n-1, (ro - (n-1) * cos(tetaRad)) / sin(tetaRad)), Scalar(255, 255, 255));
		}
		else
		{
			line(img, Point((ro - sin(tetaRad)) / cos(tetaRad), 1), Point((ro - (m-1) * sin(tetaRad)) / cos(tetaRad), m-1), Scalar(255, 255, 255));

		}
	}

	imshow("linii1", img);
	waitKey();
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
				char* filename = (char*) calloc(12,sizeof(char));
				scanf("%s", filename);
				lab1_afisare_puncte(filename);
				break;
			}
			case 11 :
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
			}
		}
	}
	while (op!=0);
	return 0;
}