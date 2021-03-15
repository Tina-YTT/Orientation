
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Marker
{
public:
	Marker();
	//~Marker();
	int id;
	vector<Point2f> points;
	static int getMarkerId(Mat &markerImage, int &nRotations);
	static int hammDistMarker(Mat bits);
	void drawContour(Mat& image, Scalar color = CV_RGB(0, 250, 0)) const;
	static Mat rotate(Mat in);
	static int mat2id(const cv::Mat &bits);

	//void findContour(cv::Mat& thresholdImg, ContoursVector& contours, int minContourPointsAllowed) const;

private:

};

Marker::Marker()
	:id(-1)
{
}

int Marker::getMarkerId(Mat& markerImage, int &nRotations)
{
	assert(markerImage.rows == markerImage.cols);
	assert(markerImage.type() == CV_8UC1);
	Mat grey = markerImage;
	threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);
	int cellSize = markerImage.rows / 7;
	for (int y = 0; y < 7; y++)
	{
		int inc = 6;
		if (y == 0 || y == 6)
			inc = 1;
		for (int x = 0; x < 7; x += inc)
		{
			int cellX = x * cellSize;
			int cellY = y * cellSize;
			Mat cell = grey(Rect(cellX, cellY, cellSize, cellSize));//以点（cellX,cellY）为起点长宽均为cellSIZE的正方形
			int nZ = countNonZero(cell);//统计正方形中非零像素的个数
			if (nZ > (cellSize*cellSize) / 2)
			{
				return -1;
			}//如果边界信息不是黑色的，就不是一个标识。
		}
	}

	Mat bitMatrix = Mat::zeros(5, 5, CV_8UC1);
	//得到信息（对于内部的网格，决定是否是黑色或白色的）就是判断内部5x5的网格都是什么颜色的，得到一个包含信息的矩阵bitMatrix。
	for (int y = 0; y < 5; y++)
	{
		for (int x = 0; x < 5; x++)
		{
			int cellX = (x + 1)*cellSize;
			int cellY = (y + 1)*cellSize;
			Mat cell = grey(Rect(cellX, cellY, cellSize, cellSize));
			int nZ = countNonZero(cell);
			if (nZ > (cellSize*cellSize) / 2)
				bitMatrix.at<uchar>(y, x) = 1;
		}
	}
	//检查所有的旋转
	Mat rotations[4];
	int distances[4];
	rotations[0] = bitMatrix;
	//cout <<"旋转次数"<< bitMatrix << endl;
	distances[0] = hammDistMarker(rotations[0]);//求没有旋转的矩阵的海明距离。
	pair<int, int> minDist(distances[0], 0);//把求得的海明距离和旋转角度作为最小初始值对，每个pair都有两个属性值first和second



	for (int i = 1; i < 4; i++) //判断这个矩阵与参考矩阵旋转多少度。
	{
		//计算最近的可能元素的海明距离
		rotations[i] = rotate(rotations[i - 1]);//每次旋转90度
		distances[i] = hammDistMarker(rotations[i]);

		if (distances[i] < minDist.first)
		{

			minDist.first = distances[i];
			minDist.second = i;//这个pair的第二个值是代表旋转几次，每次90度。
		}
	}

	nRotations = minDist.second;//这个是将返回的旋转角度值
	//cout <<"旋转次数="<< minDist.second << endl;
	if (minDist.first == 0) //若海明距离为0,则根据这个旋转后的矩阵计算标识ID
	{

		return mat2id(rotations[minDist.second]);
	}
	return -1;
}

int Marker::hammDistMarker(Mat bits)
{
	int ids[4][5] = {
			{ 1, 0, 0, 0, 0 },
			{ 1, 0, 1, 1, 1 },
			{ 0, 1, 0, 0, 1 },
			{ 0, 1, 1, 1, 0 }


	};

	int dist = 0;
	for (int y = 0; y < 5; y++)
	{
		int minSum = 1e5;
		for (int p = 0; p < 4; p++)
		{
			int sum = 0;
			for (int x = 0; x < 5; x++)
			{
				sum += bits.at<uchar>(y, x) == ids[p][x] ? 0 : 1;
			}
			if (minSum > sum)
				minSum = sum;
		}
		dist += minSum;
	}
	return dist;
}

void Marker::drawContour(Mat& image, Scalar color) const//在图像上画线，描绘出轮廓。
{
	float thickness = 2;

	line(image, points[0], points[1], color, thickness, CV_AA);
	line(image, points[1], points[2], color, thickness, CV_AA);
	line(image, points[2], points[3], color, thickness, CV_AA);//thickness线宽
	line(image, points[3], points[0], color, thickness, CV_AA);//CV_AA是抗锯齿
}

Mat Marker::rotate(Mat in)//就是把矩阵旋转90度
{
	Mat out;
	in.copyTo(out);
	for (int i = 0; i < in.rows; i++)
	{
		for (int j = 0; j < in.cols; j++)
		{
			out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);//at<uchar>用来指定某个位置的像素，同时指定数据类型。就是交换元素，怎么交换的？
		}
	}
	return out;
}

int Marker::mat2id(const Mat& bits)//移位，求或，再移位，得到最终的ID
{
	int val = 0;
	for (int y = 0; y < 5; y++)
	{
		val <<= 1;//移位操作
		if (bits.at<uchar>(y, 1)) val |= 1;
		val <<= 1;
		if (bits.at<uchar>(y, 3)) val |= 1;
	}
	return val;
}
