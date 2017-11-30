#include<opencv2\opencv.hpp>
#include<stdio.h>
#include<math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

void DFT(Mat,Mat&);//DFT变换函数
float SUM(Mat,int);//Mat对象行向量求和
int main()
{
	Mat img1 = imread("HW1.bmp",0);
	imshow("original",img1);
	
	//傅里叶变换
	Mat img2;
	DFT(img1,img2);
	imshow("离散傅里叶变换",img2);
	Mat DFT_changed;
	img2.convertTo(DFT_changed,CV_8UC1,255,0);
	imwrite("离散傅里叶变换.jpg",DFT_changed);

	//二值化
	Mat img3 = img2.clone();
	threshold(img2,img3,0.43,1,CV_THRESH_BINARY);
	imshow("二值化",img3);
	Mat Binarization;
	img3.convertTo(Binarization,CV_8UC1,255,0);
	imwrite("二值化.jpg",Binarization);

	
	//radon变换
	int length=img3.cols;
	int width = img3.rows;
	Mat img4;
	Mat rotation;
	Mat radon = Mat(256,180,CV_32FC1);//用于存储radon变换结果的矩阵
	for(int i=0;i<180;i++)
	{
		rotation= getRotationMatrix2D(Point2f(length/2,width/2),i,1.0);
		warpAffine(img3,img4,rotation,Size(length,width));//将图像旋转i角度
		
		for(int j=0;j<256;j++)
		{
			radon.at<float>(j,i) = SUM(img4,j);//求出旋转角度之后的radon值
		}
	}

	normalize(radon,radon,0,1,CV_MINMAX);
	imshow("radon变换",radon);
	Mat Radon;
	radon.convertTo(Radon,CV_8UC1,255,0);
	imwrite("radon变换.jpg",Radon);

	//找出radon变换中最大的灰度值，其对应的角度与运动模糊角度互余
	int k=0;
	float max=0;
	for(int i=0;i<radon.rows;i++)
	{
		float* p= radon.ptr<float>(i);
		for(int j=0;j<radon.cols;j++)
		{
			if(max<p[j])
			{
				max=p[j];
				k=j;
			}
		}
	}
	cout<<"运动模糊方向为"<<endl<<90-k<<endl;

	waitKey();
	return 0;

}
float SUM(Mat img,int j)
{
	float sum =0;
	for(int i =0;i<img.cols;i++)
	{
		sum+=(float)img.ptr<float>(j)[i];
	}
	return sum;
}
void DFT(Mat img, Mat &mag)
{
	
	int M = getOptimalDFTSize( img.rows );
    int N = getOptimalDFTSize( img.cols );
    Mat padded;//将原图像的大小变为m*n的大小，补充的位置填0，
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));//这里是获取了两个mat，一个用于存放dft变换的实部，一个用于存放虚部，初始的时候，实部就是图像本身，虚部全为0
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;//将几个单通道的mat融合成一个多通道的mat，这里融合的complexImg即有实部，又有虚部
    merge(planes, 2, complexImg);//dft变换，因为complexImg本身就是两个通道的mat，所以dft变换的结果也可以保存在其中
    dft(complexImg, complexImg);    //将complexImg重新拆分成两个mat，一个是实部，一个是虚部
    split(complexImg, planes);

    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    //这一部分是为了计算dft变换后的幅值，以便于显示幅值的计算公式如上
    magnitude(planes[0], planes[1], planes[0]);//将两个mat对应位置相乘
    mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);

    //修剪频谱，如果图像的行或者列是奇数的话，那其频谱是不对称的，因此要修剪
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

    int cx = mag.cols/2;
    int cy = mag.rows/2;

    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(mag, mag, 0, 1, CV_MINMAX);
}