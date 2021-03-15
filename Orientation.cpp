#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <iomanip>
#include "marker.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

#define PI 3.1416
typedef vector<Point> PointsVector;
typedef vector<PointsVector> ContoursVector;
vector<Marker> markers;
vector<Point2f> m_markerCorners2d;
vector<Point3f> m_markerCorners3d;
Size markerSize = Size(100, 100);

void findCandidates(const ContoursVector& contours, vector<Marker>& detectedMarkers);
float perimeter(const vector<Point2f> &a);//求多边形周长
void recognizeMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers);
int direction(vector<int> id, vector<double> area);
int degree();
void estimatePosition(vector<int> id, vector<Marker>& detectedMarkers, Mat_<float>& camMatrix, Mat_<float>& distCoeff);

float location(vector<Marker> a);
int g_flag = 0;
vector<int> g_id;
vector<double> g_area;
Mat g_src;
//FileStorage result("result5.yml", FileStorage::WRITE);


int main(int argc, char *argv[])
{
	/*if (!result.isOpened())
	{
		cout << "文件创建失败" << endl;
		std::system("PAUSE");
		exit(0);
	}*/
	//g_src = imread("23.jpg");
	//cout << g_src.size() << endl;
	Mat grayscale, threshImg;
	//vector<vector<Point>> contours;
	ContoursVector myContour, allContours;

	m_markerCorners2d.push_back(Point2f(0, 0));
	m_markerCorners2d.push_back(Point2f(99, 0));
	m_markerCorners2d.push_back(Point2f(99, 99));
	m_markerCorners2d.push_back(Point2f(0, 99));

	m_markerCorners3d.push_back(Point3f(-12.5f, -12.5f, 0));
	m_markerCorners3d.push_back(Point3f(+12.5f, -12.5f, 0));
	m_markerCorners3d.push_back(Point3f(+12.5f, +12.5f, 0));
	m_markerCorners3d.push_back(Point3f(-12.5f, +12.5f, 0));



	Mat_<float> intrinsMatrix = Mat::eye(3, 3, CV_64F);
	Mat_<float> distCoeff = Mat::zeros(5, 1, CV_64F);
	intrinsMatrix = (Mat_<double>(3, 3) << 1615.05133, 0, 659.10946, 0, 1615.38078, 375.54567, 0, 0, 1);
	distCoeff = (Mat_<double>(5, 1) << 0.13519, -1.89350, 0.00000, 0.00000, 0.00000);

	/*FileStorage fs("out_camera_data.yml", FileStorage::READ);
	if (!fs.isOpened())
	{
	cout << "Could not open the configuration file!" << endl;
	exit(1);
	}
	fs["Camera_Matrix"] >> intrinsMatrix;
	fs["Distortion_Coefficients"] >> distCoeff;
	fs.release();*/
	//cout << intrinsMatrix << endl;
	//cout << distCoeff << endl;

	VideoCapture capture("4.mp4");
	int flag = 0;
	while (1)
	{

		Mat src;
		capture >> g_src;
		flag++;
		//cout << g_src.size() << endl;
	if (g_src.empty())
	{
		cerr << "ERROR: could not grab a camera frame!" << endl;
		exit(1);
	}
	if (flag == (int)3000 / 36)
	{
		imshow("当前帧", g_src);

		/*Mat imageConvert;
		g_src.convertTo(imageConvert, g_src.type(), 1.5,-200);//调对比度和亮度
		imwrite("对比度亮度.png", imageConvert);*/
	cvtColor(g_src, grayscale, CV_BGR2GRAY);
	//imwrite("灰度.png", grayscale);
	//adaptiveThreshold(grayscale, threshImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	threshold(grayscale, threshImg, 80, 255, THRESH_BINARY_INV);

	//cvWaitKey(0);
	//imwrite("Threshold image.png", threshImg);
	/*检测所输入的二值图像的轮廓，返回一个多边形列表，其每个多边形标识一个轮廓，小轮廓不关注，不包括标记*/
	findContours(threshImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	myContour.clear();
	for (size_t i = 0; i < allContours.size(); i++)
	{
		int contourSize = allContours[i].size();
		if (contourSize > grayscale.cols / 5)
		{
			myContour.push_back(allContours[i]);
		}
	}
	Mat contoursImage(threshImg.size(), CV_8UC1);
	contoursImage = Scalar(0);
	drawContours(contoursImage, myContour, -1, cv::Scalar(255), 2);
	//imshow("Contours", contoursImage);
	//imwrite("Contours.png", contoursImage);
	findCandidates(myContour, markers);
	recognizeMarkers(grayscale, markers);
	estimatePosition(g_id, markers, intrinsMatrix, distCoeff);
	flag = 0;
	g_id.clear();
	}

	waitKey(36);

	}

	system("PAUSE");
	return 0;
}

/*由于我们的标记是四边形，当找到图像所有轮廓细节后，本文用Opencv内置API检测多边形，
通过判断多边形定点数量是否为4，四边形各顶点之间相互距离是否满足要求(四边形是否足够大)，过滤非候选区域。
然后再根据候选区域之间距离进一步筛选，得到最终的候选区域，并使得候选区域的顶点坐标逆时针排列。*/
void findCandidates(const ContoursVector& contours, vector<Marker>& detectedMarkers)
{
	vector<Point> approxCurve;//返回结果为多边形，用点集表示//相似形状
	vector<Marker> possibleMarkers;//可能的标记

	for (size_t i = 0; i < contours.size(); i++)
	{
		/*近似一个多边形逼近，为了减少轮廓的像素。这样比较好，可筛选出非标记区域，因为标记总能被四个顶点的多边形表示。
		如果多边形的顶点多于或少于四个，就绝对不是本项目想要的标记。通过点集近似多边形，第三个参数为epsilon代表近似程度，
		即原始轮廓及近似多边形之间的距离，第四个参数表示多边形是闭合的。*/
		double eps = contours[i].size()*0.05;
		/*输入图像的2维点集，输出结果，估计精度，是否闭合。输出多边形的顶点组成的点集*/
		//使多边形边缘平滑，得到近似的多边形
		approxPolyDP(contours[i], approxCurve, eps, true);

		//我们感兴趣的多边形只有四个顶点
		if (approxCurve.size() != 4)
			continue;

		//检查轮廓是否是凸边形
		if (!isContourConvex(approxCurve))
			continue;
		//cout << "四边形" << approxCurve.size() << endl;
		//确保相邻的两点间的距离“足够大”－大到是一条边而不是短线段就是了
		//float minDist = numeric_limits<float>::max();//代表float可以表示的最大值，numeric_limits就是模板类，这里表示max（float）;3.4e038
		float minDist = 1e10;
		float maxDist = 0.0;
		//求当前四边形各顶点之间的最短距离
		for (int i = 0; i < 4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];//两个点坐标相减，得到向量
			float squaredSideLength = side.dot(side);//求2维向量的点积，就是XxY=x^2*y^2
			minDist = min(minDist, squaredSideLength);//找出最小的距离
		}

		for (int i = 0; i < 4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];//两个点坐标相减，得到向量
			float squaredSideLength = side.dot(side);//求2维向量的点积，就是XxY=x^2*y^2
			maxDist = max(maxDist, squaredSideLength);//找出最小的距离
		}

		//检查距离是不是特别小，小的话就退出本次循环，开始下一次循环
		if (minDist < (g_src.rows / 20)*(g_src.cols / 20))/////////////////////////////////////////////////////******
			continue;
		//cout << "1" << endl;
		if (maxDist > (g_src.rows / 4)*(g_src.cols / 3))/////////////////////////////////////////////////////******
			continue;
		//cout << "11" << endl;
		/*所有的测试通过了，保存标识候选，当四边形大小合适，则将该四边形maker放入possibleMarkers容器内 */

		Marker m;
		for (int i = 0; i < 4; i++)
			m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));//vector头文件里面就有这个push_back函数，在vector类中作用为在vector尾部加入一个数据。

		/*逆时针保存这些点
		marker中的点集本来就两种序列：顺时针和逆时针，这里要把顺时针的序列改成逆时针，
		在多边形逼近时，多边形是闭合的，则不是顺时针就是逆时针*/
		/*在第一个和第二个点之间跟踪出一条线，如果第三个点在右边，则点是逆时针保存的//逆时针排列这些点,第一个点和第二个点之间连一条线,如果第三个点在边，那么这些点就是逆时针*/
		Point v1 = m.points[1] - m.points[0];
		Point v2 = m.points[2] - m.points[0];

		/*行列式的几何意义有两个解释：一个解释是行列式就是行列式中的行或列向量所构成的超平行多面体的有向面积或有向体积；
		另一个解释是矩阵A的行列式detA就是线性变换A下的图形面积或体积的伸缩因子。
		以行向量a=(a1,a2)，b=(b1,b2)为邻边的平行四边形的有向面积：
		若这个平行四边形是由向量沿逆时针方向转到b而得到的，面积取正值；
		若这个平行四边形是由向量a沿顺时针方向转到而得到的，面积取负值； */
		double o = (v1.x*v2.y) - (v1.y*v2.x);
		if (o < 0.0)//如果第三个点在左边，那么交换第一个点和第三个点，逆时针保存
			swap(m.points[1], m.points[3]);
		possibleMarkers.push_back(m);//把这个标识放入候选标识向量中
	}

	/*移除那些角点互相离的太近的四边形*/
	vector< pair<int, int > > tooNearCandidates;
	cout << "可能的" << possibleMarkers.size() << endl;
	/*计算两个maker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近,
	把这一对四边形放入移除队列。//计算每个边角到其他可能标记的最近边角的平均距离*/
	for (size_t i = 0; i < possibleMarkers.size(); i++) {
		const Marker& m1 = possibleMarkers[i];
		for (size_t j = i + 1; j < possibleMarkers.size(); j++) {
			const Marker& m2 = possibleMarkers[j];
			float distSquared = 0.0;
			for (int c = 0; c < 4; c++) {
				Point v = m1.points[c] - m2.points[c];
				distSquared = v.dot(v);
			}
			distSquared /= 4;

			if (distSquared < 100)
				tooNearCandidates.push_back(pair<int, int>(i, j));
		}
	}

	//移除了相邻的元素对的标识
	//计算距离相近的两个marker内部，四个点的距离和，将距离和较小的，在removlaMask内做标记，即不作为最终的detectedMarkers 
	vector<bool> removalMask(possibleMarkers.size(), false);
	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		//求这一对相邻四边形的周长
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

		//谁周长小，移除谁
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		removalMask[removalIndex] = true;
	}

	//返回候选，移除相邻四边形中周长较小的那个，放入待检测的四边形的队列中。//返回可能的对象
	detectedMarkers.clear();
	for (size_t i = 0; i < possibleMarkers.size(); i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}

}

float perimeter(const vector<Point2f> &a)//求多边形周长。
{
	float sum = 0, dx, dy;
	for (size_t i = 0; i < a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy * dy);
	}

	return sum;
}

void recognizeMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers)
{
	Mat canonicalMarkerImage;
	vector<Marker> goodMarkers;
	//cout <<"标记矩阵个数"<< detectedMarkers.size() << endl;
	int n = 0;
	/*Identify the markers识别标识 //分析每一个捕获到的标记，去掉透视投影，得到平面／正面的矩形。
	//为了得到这些矩形的标记图像，我们不得不使用透视变换去恢复(unwarp)输入的图像。
	这个矩阵应该使用cv::getPerspectiveTransform函数，它首先根据四个对应的点找到透视变换，
	第一个参数是标记的坐标，第二个是正方形标记图像的坐标。估算的变换将会把标记转换成方形，从而方便我们分析。 */

	for (int i = 0; i < (detectedMarkers.size() - n); i++)
	{

		Marker& marker = detectedMarkers[i];
		//找到透视转换矩阵，获得矩形区域的正面视图
		// 找到透视投影，并把标记转换成矩形，输入图像四边形顶点坐标，输出图像的相应的四边形顶点坐标 
		Mat markerTransform = getPerspectiveTransform(marker.points, m_markerCorners2d);//输入原始图像和变换之后的图像的对应4个点，便可以得到变换矩阵

		/* Transform image to get a canonical marker image
		// Transform image to get a canonical marker image
		//输入的图像
		//输出的图像
		//3x3变换矩阵 */
		warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);
		//对图像进行透视变换,这就得到和标识图像一致正面的图像，方向可能不同，看四个点如何排列的了。感觉这个变换后，就得到只有标识图的正面图
		threshold(canonicalMarkerImage, canonicalMarkerImage, 70, 255, THRESH_BINARY | THRESH_OTSU);
		imwrite("canonicalMarkerImage.png", canonicalMarkerImage);

		Mat markerImage = grayscale.clone();
		marker.drawContour(markerImage);
		Mat markerSubImage = markerImage(boundingRect(marker.points));//boundingRect计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的


		char t[30]; sprintf_s(t, "sourcemarker%d.png", i);
		imwrite(t, markerSubImage);


		char l[30]; sprintf_s(l, "Marker%d.png", i);
		imwrite(l, canonicalMarkerImage);


		int nRotations;
		int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
		if (id == -1)
		{
			n++;
			for (int k = i; k < (detectedMarkers.size() - n); k++)
				detectedMarkers[k] = detectedMarkers[k + 1];
			i--;
			continue;
		}
		g_id.push_back(id);

		cout << "ID = " << g_id[i] << endl;
		//result << "ID" << g_id[i];
		/*if (g_id[i] == -1)
		continue;*/

		if (g_id[i] != -1)
		{
			marker.id = g_id[i];
			//根据相机的旋转，调整标记的姿态
			rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());//循环移位
			goodMarkers.push_back(marker);
		}
	}

	//把所有标识的四个顶点都放在一个大的向量中。
	if (goodMarkers.size() > 0)
	{
		//找到所有标记的角点
		vector<Point2f> preciseCorners(4 * goodMarkers.size());//每个marker四个点
		for (size_t i = 0; i < goodMarkers.size(); i++) {
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++)
			{
				preciseCorners[i * 4 + c] = marker.points[c];//i表示第几个marker，c表示某个marker的第几个点
			}
		}

		TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 0.01);
		//这个是迭代终止条件，这里是达到30次迭代或者达到0.01精度终止。角点精准化迭代过程的终止条件
		cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);
		//发现亚像素精度的角点位置，第二个参数代表输入的角点的初始位置并输出精准化的坐标。
		//在标记检测的早期的阶段没有使用cornerSubPix函数是因为它的复杂性－调用这个函数处理大量顶点时会耗费大量的处理时间，因此我们只在处理有效标记时使用。

		for (size_t i = 0; i < goodMarkers.size(); i++)  //再把精准化的坐标传给每一个标识。// 保存最新的顶点
		{
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++) {
				marker.points[c] = preciseCorners[i * 4 + c];
			}
		}
	}

	//画出细化后的矩形图片
	Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = Scalar(0);



	for (size_t i = 0; i < goodMarkers.size(); i++)
	{
		goodMarkers[i].drawContour(markerCornersMat, Scalar(255));

		g_area.push_back(contourArea(goodMarkers[i].points, true));//求每个图形的面积
		//cout << "面积" << i << "=" << contourArea(goodMarkers[i].points, true) << endl;

	}
	//direction(g_id, g_area);
	imwrite("refine.jpg", grayscale*0.5 + markerCornersMat);
	detectedMarkers = goodMarkers;
	/*float location1;
	for (int i = 0; i < detectedMarkers.size(); i++)
	{
	location1=location(detectedMarkers);
	}
	cout << "距离=" << location1 <<"米"<< endl;
	*/

}
/*
int direction(vector<int> id, vector<double> area)
{
int degree();

if (id.size() == 1)
{
if (id[0] == 213)
{
cout << "正北" << endl;
g_flag = 1;

}
if (id[0] == 978)
{
cout << "正南" << endl;
g_flag = 2;

}
if (id[0] == 29)
{
cout << "正东" << endl;
g_flag = 3;

}
if (id[0] == 710)
{
cout << "正西" << endl;
g_flag = 4;

}
}
else if (id.size() == 2)
{

if ((id[0] == 213 && id[1] == 29) || (id[0] == 29 && id[1] == 213))
{
cout << "北偏东" << degree() << endl;
g_flag = 5;

}
if ((id[0] == 213 && id[1] == 710) || (id[0] == 710 && id[1] == 213))
{
cout << "北偏西" <<  degree() << endl;
g_flag = 6;

}
if ((id[0] == 978 && id[1] == 29) || (id[0] == 29 && id[1] == 978))
{
cout << "南偏东" << degree() << endl;
g_flag = 7;

}
if ((id[0] == 978 && id[1] == 710) || (id[0] == 710 && id[1] == 978))
{
cout << "南偏西" << degree() << endl;
g_flag = 8;

}

}
return g_flag;

}



int degree()
{

int degree = 0;
vector<float> area;
for (int i = 0; i < g_area.size(); i++)
{
area.push_back( g_area[i]);
}

float change;
if (g_id[0] ==29|| g_id[0] ==710)
{
change = area[1];
area[1] = area[0];
area[0] = change;

}



degree=atan(area[0] /area[1]) * 180/PI;


return degree;
}*/

/*float location(vector<Marker> a)
{
float L0 = 0.0;
float L1 = 0.0;
float location;
if (g_flag == 1 || g_flag == 2 || g_flag == 3 || g_flag == 4)
{
L0 = perimeter(a[0].points);
}
else if (g_flag == 5 || g_flag == 6 || g_flag == 7 || g_flag == 8)
{
L0 = perimeter(a[0].points);
L1 = perimeter(a[1].points);
}
if(L0>=L1)
{
location=380/L0;
}
else if(L0<L1)
{
location=380/L0;
}
return location;
}*/

void estimatePosition(vector<int> id, vector<Marker>& detectedMarkers, Mat_<float>& camMatrix, Mat_<float>& distCoeff)
{

	int angle1;
	float dist1;
	vector<float> x, z, d;
	float theta_z;
	float theta_y;
	float theta_x;
	//vector<int> angle;

	for (size_t i = 0; i < detectedMarkers.size(); i++)
	{

		Marker& m = detectedMarkers[i];

		Mat Rvec;
		Mat_<float> Tvec;//Mat_<float>对应的是CV_32F
		Mat raux, taux;
		solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
		raux.convertTo(Rvec, CV_32F);//转换Mat的保存类型，输出Rvec
		taux.convertTo(Tvec, CV_32F);
		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);//罗德里格斯变换对旋转向量和旋转矩阵进行转换，输出旋转矩阵
		/*cout << "旋转矩阵：\n" << rotMat << endl
		<< "平移矩阵：\n" << Tvec << endl;*/

		theta_z = atan2(rotMat[1][0], rotMat[0][0]) * 180 / PI;
		theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 180 / PI;
		theta_x = atan2(rotMat[2][1], rotMat[2][2]) * 180 / PI;
		/*cout << "α=" << theta_x << endl;
		cout << "baita=" << theta_y << endl;
		cout << "γ=" << theta_z << endl;*/

		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R_n;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> T_n;
		cv2eigen(rotMat, R_n);
		cv2eigen(Tvec, T_n);
		Eigen::Vector3f P_oc;

		P_oc = -R_n.inverse()*T_n;
		//angle1 = atan(  abs(P_oc[1]) / abs(P_oc[0])   ) * 180 / PI;
		//angle.push_back(angle1);
		x.push_back(P_oc[0]);
		z.push_back(P_oc[2]);
		float dist = sqrt((abs(T_n(0)))*(abs(T_n(0))) + (abs(T_n(1)))*(abs(T_n(1))) + (abs(T_n(2)))*(abs(T_n(2))));
		d.push_back(dist);

		//cout << "世界坐标" << P_oc << endl;

	}

	if (id.size() == 1)
	{
		if (x[0] < 0)
		{
			if (id[0] == 213)
			{
				cout << "北偏东" << int(abs(theta_y)) << "°" << endl;
				//result << "north-east" << int(abs(theta_y));
				g_flag = 1;
			}
			if (id[0] == 29)
			{
				cout << "南偏东" << int(90 - abs(theta_y)) << "°" << endl;
				//result << "south-east" << int(90 - abs(theta_y));
				g_flag = 2;
			}
			if (id[0] == 623)
			{
				cout << "南偏西" << int(abs(theta_y)) << "°" << endl;
				//result << "south-west" << int(abs(theta_y));
				g_flag = 3;
			}
			if (id[0] == 809)
			{
				cout << "北偏西" << int(90 - abs(theta_y)) << "°" << endl;
				//result << "north-west" << int(90 - abs(theta_y));
				g_flag = 4;
			}

		}
		if (x[0] >= 0)
		{
			if (id[0] == 213)
			{
				cout << "北偏西" << int(abs(theta_y)) << "°" << endl;
				//result << "north-west" << int(abs(theta_y));
				g_flag = 1;
			}
			if (id[0] == 29)
			{
				cout << "北偏东" << int(90 - abs(theta_y)) << "°" << endl;
				//result << "north-east" << int(90 - abs(theta_y));
				g_flag = 2;
			}
			if (id[0] == 623)
			{
				cout << "南偏东" << int(abs(theta_y)) << "°" << endl;
				//result << "south-east" << int(abs(theta_y));
				g_flag = 3;
			}
			if (id[0] == 809)
			{
				cout << "南偏西" << int(90 - abs(theta_y)) << "°" << endl;
				//result << "sorth-west" << int(90 - abs(theta_y));
				g_flag = 4;
			}
		}

	}
	else if (id.size() == 2)
	{

		if ((id[0] == 213 && id[1] == 29) || (id[0] == 29 && id[1] == 213))
		{
			cout << "北偏东";
			//result << "north-east";

			g_flag = 5;

		}
		if ((id[0] == 213 && id[1] == 809) || (id[0] == 809 && id[1] == 213))
		{
			cout << "北偏西";
			//result << "north-west";
			g_flag = 6;

		}
		if ((id[0] == 623 && id[1] == 29) || (id[0] == 29 && id[1] == 623))
		{
			cout << "南偏东";
			//result << "south-east";
			g_flag = 7;

		}
		if ((id[0] == 623 && id[1] == 809) || (id[0] == 809 && id[1] == 623))
		{
			cout << "南偏西";
			//result << "south-west";
			g_flag = 8;

		}
	}


	if (g_flag == 1 || g_flag == 2 || g_flag == 3 || g_flag == 4)
	{
		dist1 = d[0];
		cout << "距沙盘中心" << int(dist1) << "cm" << endl;
		//result << "distance" << int(dist1);
	}

	if (g_flag == 5 || g_flag == 6 || g_flag == 7 || g_flag == 8)
	{
		if (g_id[0] == 29 || g_id[0] == 809)
			angle1 = atan((abs(z[0])) / (abs(z[1]))) * 180 / PI;

		if (g_id[0] == 623 || g_id[0] == 213)
			angle1 = atan((abs(z[1])) / (abs(z[0]))) * 180 / PI;
		cout << angle1 << "°" << endl;
		//result << angle1;

		if (d[0] <= d[1])
		{
			dist1 = d[0];
		}
		else if (d[0] > d[1])
		{
			dist1 = d[1];
		}

		cout << "距沙盘中心" << int(dist1) << "cm" << endl;
		//result << "distance" << int(dist1);

	}
	g_flag = 0;

}