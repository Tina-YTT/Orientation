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
float perimeter(const vector<Point2f> &a);//�������ܳ�
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
		cout << "�ļ�����ʧ��" << endl;
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
		imshow("��ǰ֡", g_src);

		/*Mat imageConvert;
		g_src.convertTo(imageConvert, g_src.type(), 1.5,-200);//���ԱȶȺ�����
		imwrite("�Աȶ�����.png", imageConvert);*/
	cvtColor(g_src, grayscale, CV_BGR2GRAY);
	//imwrite("�Ҷ�.png", grayscale);
	//adaptiveThreshold(grayscale, threshImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	threshold(grayscale, threshImg, 80, 255, THRESH_BINARY_INV);

	//cvWaitKey(0);
	//imwrite("Threshold image.png", threshImg);
	/*���������Ķ�ֵͼ�������������һ��������б���ÿ������α�ʶһ��������С��������ע�����������*/
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

/*�������ǵı�����ı��Σ����ҵ�ͼ����������ϸ�ں󣬱�����Opencv����API������Σ�
ͨ���ж϶���ζ��������Ƿ�Ϊ4���ı��θ�����֮���໥�����Ƿ�����Ҫ��(�ı����Ƿ��㹻��)�����˷Ǻ�ѡ����
Ȼ���ٸ��ݺ�ѡ����֮������һ��ɸѡ���õ����յĺ�ѡ���򣬲�ʹ�ú�ѡ����Ķ���������ʱ�����С�*/
void findCandidates(const ContoursVector& contours, vector<Marker>& detectedMarkers)
{
	vector<Point> approxCurve;//���ؽ��Ϊ����Σ��õ㼯��ʾ//������״
	vector<Marker> possibleMarkers;//���ܵı��

	for (size_t i = 0; i < contours.size(); i++)
	{
		/*����һ������αƽ���Ϊ�˼������������ء������ȽϺã���ɸѡ���Ǳ��������Ϊ������ܱ��ĸ�����Ķ���α�ʾ��
		�������εĶ�����ڻ������ĸ����;��Բ��Ǳ���Ŀ��Ҫ�ı�ǡ�ͨ���㼯���ƶ���Σ�����������Ϊepsilon������Ƴ̶ȣ�
		��ԭʼ���������ƶ����֮��ľ��룬���ĸ�������ʾ������Ǳպϵġ�*/
		double eps = contours[i].size()*0.05;
		/*����ͼ���2ά�㼯�������������ƾ��ȣ��Ƿ�պϡ��������εĶ�����ɵĵ㼯*/
		//ʹ����α�Եƽ�����õ����ƵĶ����
		approxPolyDP(contours[i], approxCurve, eps, true);

		//���Ǹ���Ȥ�Ķ����ֻ���ĸ�����
		if (approxCurve.size() != 4)
			continue;

		//��������Ƿ���͹����
		if (!isContourConvex(approxCurve))
			continue;
		//cout << "�ı���" << approxCurve.size() << endl;
		//ȷ�����ڵ������ľ��롰�㹻�󡱣�����һ���߶����Ƕ��߶ξ�����
		//float minDist = numeric_limits<float>::max();//����float���Ա�ʾ�����ֵ��numeric_limits����ģ���࣬�����ʾmax��float��;3.4e038
		float minDist = 1e10;
		float maxDist = 0.0;
		//��ǰ�ı��θ�����֮�����̾���
		for (int i = 0; i < 4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];//����������������õ�����
			float squaredSideLength = side.dot(side);//��2ά�����ĵ��������XxY=x^2*y^2
			minDist = min(minDist, squaredSideLength);//�ҳ���С�ľ���
		}

		for (int i = 0; i < 4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];//����������������õ�����
			float squaredSideLength = side.dot(side);//��2ά�����ĵ��������XxY=x^2*y^2
			maxDist = max(maxDist, squaredSideLength);//�ҳ���С�ľ���
		}

		//�������ǲ����ر�С��С�Ļ����˳�����ѭ������ʼ��һ��ѭ��
		if (minDist < (g_src.rows / 20)*(g_src.cols / 20))/////////////////////////////////////////////////////******
			continue;
		//cout << "1" << endl;
		if (maxDist > (g_src.rows / 4)*(g_src.cols / 3))/////////////////////////////////////////////////////******
			continue;
		//cout << "11" << endl;
		/*���еĲ���ͨ���ˣ������ʶ��ѡ�����ı��δ�С���ʣ��򽫸��ı���maker����possibleMarkers������ */

		Marker m;
		for (int i = 0; i < 4; i++)
			m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));//vectorͷ�ļ�����������push_back��������vector��������Ϊ��vectorβ������һ�����ݡ�

		/*��ʱ�뱣����Щ��
		marker�еĵ㼯�������������У�˳ʱ�����ʱ�룬����Ҫ��˳ʱ������иĳ���ʱ�룬
		�ڶ���αƽ�ʱ��������Ǳպϵģ�����˳ʱ�������ʱ��*/
		/*�ڵ�һ���͵ڶ�����֮����ٳ�һ���ߣ���������������ұߣ��������ʱ�뱣���//��ʱ��������Щ��,��һ����͵ڶ�����֮����һ����,������������ڱߣ���ô��Щ�������ʱ��*/
		Point v1 = m.points[1] - m.points[0];
		Point v2 = m.points[2] - m.points[0];

		/*����ʽ�ļ����������������ͣ�һ������������ʽ��������ʽ�е��л������������ɵĳ�ƽ�ж������������������������
		��һ�������Ǿ���A������ʽdetA�������Ա任A�µ�ͼ�������������������ӡ�
		��������a=(a1,a2)��b=(b1,b2)Ϊ�ڱߵ�ƽ���ı��ε����������
		�����ƽ���ı���������������ʱ�뷽��ת��b���õ��ģ����ȡ��ֵ��
		�����ƽ���ı�����������a��˳ʱ�뷽��ת�����õ��ģ����ȡ��ֵ�� */
		double o = (v1.x*v2.y) - (v1.y*v2.x);
		if (o < 0.0)//���������������ߣ���ô������һ����͵������㣬��ʱ�뱣��
			swap(m.points[1], m.points[3]);
		possibleMarkers.push_back(m);//�������ʶ�����ѡ��ʶ������
	}

	/*�Ƴ���Щ�ǵ㻥�����̫�����ı���*/
	vector< pair<int, int > > tooNearCandidates;
	cout << "���ܵ�" << possibleMarkers.size() << endl;
	/*��������maker�ı���֮��ľ��룬�����֮�����͵�ƽ��ֵ����ƽ��ֵ��С������Ϊ����maker�����,
	����һ���ı��η����Ƴ����С�//����ÿ���߽ǵ��������ܱ�ǵ�����߽ǵ�ƽ������*/
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

	//�Ƴ������ڵ�Ԫ�ضԵı�ʶ
	//����������������marker�ڲ����ĸ���ľ���ͣ�������ͽ�С�ģ���removlaMask������ǣ�������Ϊ���յ�detectedMarkers 
	vector<bool> removalMask(possibleMarkers.size(), false);
	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		//����һ�������ı��ε��ܳ�
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

		//˭�ܳ�С���Ƴ�˭
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		removalMask[removalIndex] = true;
	}

	//���غ�ѡ���Ƴ������ı������ܳ���С���Ǹ�������������ı��εĶ����С�//���ؿ��ܵĶ���
	detectedMarkers.clear();
	for (size_t i = 0; i < possibleMarkers.size(); i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}

}

float perimeter(const vector<Point2f> &a)//�������ܳ���
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
	//cout <<"��Ǿ������"<< detectedMarkers.size() << endl;
	int n = 0;
	/*Identify the markersʶ���ʶ //����ÿһ�����񵽵ı�ǣ�ȥ��͸��ͶӰ���õ�ƽ�棯����ľ��Ρ�
	//Ϊ�˵õ���Щ���εı��ͼ�����ǲ��ò�ʹ��͸�ӱ任ȥ�ָ�(unwarp)�����ͼ��
	�������Ӧ��ʹ��cv::getPerspectiveTransform�����������ȸ����ĸ���Ӧ�ĵ��ҵ�͸�ӱ任��
	��һ�������Ǳ�ǵ����꣬�ڶ����������α��ͼ������ꡣ����ı任����ѱ��ת���ɷ��Σ��Ӷ��������Ƿ����� */

	for (int i = 0; i < (detectedMarkers.size() - n); i++)
	{

		Marker& marker = detectedMarkers[i];
		//�ҵ�͸��ת�����󣬻�þ��������������ͼ
		// �ҵ�͸��ͶӰ�����ѱ��ת���ɾ��Σ�����ͼ���ı��ζ������꣬���ͼ�����Ӧ���ı��ζ������� 
		Mat markerTransform = getPerspectiveTransform(marker.points, m_markerCorners2d);//����ԭʼͼ��ͱ任֮���ͼ��Ķ�Ӧ4���㣬����Եõ��任����

		/* Transform image to get a canonical marker image
		// Transform image to get a canonical marker image
		//�����ͼ��
		//�����ͼ��
		//3x3�任���� */
		warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);
		//��ͼ�����͸�ӱ任,��͵õ��ͱ�ʶͼ��һ�������ͼ�񣬷�����ܲ�ͬ�����ĸ���������е��ˡ��о�����任�󣬾͵õ�ֻ�б�ʶͼ������ͼ
		threshold(canonicalMarkerImage, canonicalMarkerImage, 70, 255, THRESH_BINARY | THRESH_OTSU);
		imwrite("canonicalMarkerImage.png", canonicalMarkerImage);

		Mat markerImage = grayscale.clone();
		marker.drawContour(markerImage);
		Mat markerSubImage = markerImage(boundingRect(marker.points));//boundingRect���������Ĵ�ֱ�߽���С���Σ���������ͼ�����±߽�ƽ�е�


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
			//�����������ת��������ǵ���̬
			rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());//ѭ����λ
			goodMarkers.push_back(marker);
		}
	}

	//�����б�ʶ���ĸ����㶼����һ����������С�
	if (goodMarkers.size() > 0)
	{
		//�ҵ����б�ǵĽǵ�
		vector<Point2f> preciseCorners(4 * goodMarkers.size());//ÿ��marker�ĸ���
		for (size_t i = 0; i < goodMarkers.size(); i++) {
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++)
			{
				preciseCorners[i * 4 + c] = marker.points[c];//i��ʾ�ڼ���marker��c��ʾĳ��marker�ĵڼ�����
			}
		}

		TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 0.01);
		//����ǵ�����ֹ�����������Ǵﵽ30�ε������ߴﵽ0.01������ֹ���ǵ㾫׼���������̵���ֹ����
		cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);
		//���������ؾ��ȵĽǵ�λ�ã��ڶ���������������Ľǵ�ĳ�ʼλ�ò������׼�������ꡣ
		//�ڱ�Ǽ������ڵĽ׶�û��ʹ��cornerSubPix��������Ϊ���ĸ����ԣ�����������������������ʱ��ķѴ����Ĵ���ʱ�䣬�������ֻ�ڴ�����Ч���ʱʹ�á�

		for (size_t i = 0; i < goodMarkers.size(); i++)  //�ٰѾ�׼�������괫��ÿһ����ʶ��// �������µĶ���
		{
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++) {
				marker.points[c] = preciseCorners[i * 4 + c];
			}
		}
	}

	//����ϸ����ľ���ͼƬ
	Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = Scalar(0);



	for (size_t i = 0; i < goodMarkers.size(); i++)
	{
		goodMarkers[i].drawContour(markerCornersMat, Scalar(255));

		g_area.push_back(contourArea(goodMarkers[i].points, true));//��ÿ��ͼ�ε����
		//cout << "���" << i << "=" << contourArea(goodMarkers[i].points, true) << endl;

	}
	//direction(g_id, g_area);
	imwrite("refine.jpg", grayscale*0.5 + markerCornersMat);
	detectedMarkers = goodMarkers;
	/*float location1;
	for (int i = 0; i < detectedMarkers.size(); i++)
	{
	location1=location(detectedMarkers);
	}
	cout << "����=" << location1 <<"��"<< endl;
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
cout << "����" << endl;
g_flag = 1;

}
if (id[0] == 978)
{
cout << "����" << endl;
g_flag = 2;

}
if (id[0] == 29)
{
cout << "����" << endl;
g_flag = 3;

}
if (id[0] == 710)
{
cout << "����" << endl;
g_flag = 4;

}
}
else if (id.size() == 2)
{

if ((id[0] == 213 && id[1] == 29) || (id[0] == 29 && id[1] == 213))
{
cout << "��ƫ��" << degree() << endl;
g_flag = 5;

}
if ((id[0] == 213 && id[1] == 710) || (id[0] == 710 && id[1] == 213))
{
cout << "��ƫ��" <<  degree() << endl;
g_flag = 6;

}
if ((id[0] == 978 && id[1] == 29) || (id[0] == 29 && id[1] == 978))
{
cout << "��ƫ��" << degree() << endl;
g_flag = 7;

}
if ((id[0] == 978 && id[1] == 710) || (id[0] == 710 && id[1] == 978))
{
cout << "��ƫ��" << degree() << endl;
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
		Mat_<float> Tvec;//Mat_<float>��Ӧ����CV_32F
		Mat raux, taux;
		solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
		raux.convertTo(Rvec, CV_32F);//ת��Mat�ı������ͣ����Rvec
		taux.convertTo(Tvec, CV_32F);
		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);//�޵����˹�任����ת��������ת�������ת���������ת����
		/*cout << "��ת����\n" << rotMat << endl
		<< "ƽ�ƾ���\n" << Tvec << endl;*/

		theta_z = atan2(rotMat[1][0], rotMat[0][0]) * 180 / PI;
		theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 180 / PI;
		theta_x = atan2(rotMat[2][1], rotMat[2][2]) * 180 / PI;
		/*cout << "��=" << theta_x << endl;
		cout << "baita=" << theta_y << endl;
		cout << "��=" << theta_z << endl;*/

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

		//cout << "��������" << P_oc << endl;

	}

	if (id.size() == 1)
	{
		if (x[0] < 0)
		{
			if (id[0] == 213)
			{
				cout << "��ƫ��" << int(abs(theta_y)) << "��" << endl;
				//result << "north-east" << int(abs(theta_y));
				g_flag = 1;
			}
			if (id[0] == 29)
			{
				cout << "��ƫ��" << int(90 - abs(theta_y)) << "��" << endl;
				//result << "south-east" << int(90 - abs(theta_y));
				g_flag = 2;
			}
			if (id[0] == 623)
			{
				cout << "��ƫ��" << int(abs(theta_y)) << "��" << endl;
				//result << "south-west" << int(abs(theta_y));
				g_flag = 3;
			}
			if (id[0] == 809)
			{
				cout << "��ƫ��" << int(90 - abs(theta_y)) << "��" << endl;
				//result << "north-west" << int(90 - abs(theta_y));
				g_flag = 4;
			}

		}
		if (x[0] >= 0)
		{
			if (id[0] == 213)
			{
				cout << "��ƫ��" << int(abs(theta_y)) << "��" << endl;
				//result << "north-west" << int(abs(theta_y));
				g_flag = 1;
			}
			if (id[0] == 29)
			{
				cout << "��ƫ��" << int(90 - abs(theta_y)) << "��" << endl;
				//result << "north-east" << int(90 - abs(theta_y));
				g_flag = 2;
			}
			if (id[0] == 623)
			{
				cout << "��ƫ��" << int(abs(theta_y)) << "��" << endl;
				//result << "south-east" << int(abs(theta_y));
				g_flag = 3;
			}
			if (id[0] == 809)
			{
				cout << "��ƫ��" << int(90 - abs(theta_y)) << "��" << endl;
				//result << "sorth-west" << int(90 - abs(theta_y));
				g_flag = 4;
			}
		}

	}
	else if (id.size() == 2)
	{

		if ((id[0] == 213 && id[1] == 29) || (id[0] == 29 && id[1] == 213))
		{
			cout << "��ƫ��";
			//result << "north-east";

			g_flag = 5;

		}
		if ((id[0] == 213 && id[1] == 809) || (id[0] == 809 && id[1] == 213))
		{
			cout << "��ƫ��";
			//result << "north-west";
			g_flag = 6;

		}
		if ((id[0] == 623 && id[1] == 29) || (id[0] == 29 && id[1] == 623))
		{
			cout << "��ƫ��";
			//result << "south-east";
			g_flag = 7;

		}
		if ((id[0] == 623 && id[1] == 809) || (id[0] == 809 && id[1] == 623))
		{
			cout << "��ƫ��";
			//result << "south-west";
			g_flag = 8;

		}
	}


	if (g_flag == 1 || g_flag == 2 || g_flag == 3 || g_flag == 4)
	{
		dist1 = d[0];
		cout << "��ɳ������" << int(dist1) << "cm" << endl;
		//result << "distance" << int(dist1);
	}

	if (g_flag == 5 || g_flag == 6 || g_flag == 7 || g_flag == 8)
	{
		if (g_id[0] == 29 || g_id[0] == 809)
			angle1 = atan((abs(z[0])) / (abs(z[1]))) * 180 / PI;

		if (g_id[0] == 623 || g_id[0] == 213)
			angle1 = atan((abs(z[1])) / (abs(z[0]))) * 180 / PI;
		cout << angle1 << "��" << endl;
		//result << angle1;

		if (d[0] <= d[1])
		{
			dist1 = d[0];
		}
		else if (d[0] > d[1])
		{
			dist1 = d[1];
		}

		cout << "��ɳ������" << int(dist1) << "cm" << endl;
		//result << "distance" << int(dist1);

	}
	g_flag = 0;

}