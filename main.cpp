#include<opencv2\opencv.hpp>
#include<stdio.h>
#include<math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

void DFT(Mat,Mat&);//DFT�任����
float SUM(Mat,int);//Mat�������������
int main()
{
	Mat img1 = imread("HW1.bmp",0);
	imshow("original",img1);
	
	//����Ҷ�任
	Mat img2;
	DFT(img1,img2);
	imshow("��ɢ����Ҷ�任",img2);
	Mat DFT_changed;
	img2.convertTo(DFT_changed,CV_8UC1,255,0);
	imwrite("��ɢ����Ҷ�任.jpg",DFT_changed);

	//��ֵ��
	Mat img3 = img2.clone();
	threshold(img2,img3,0.43,1,CV_THRESH_BINARY);
	imshow("��ֵ��",img3);
	Mat Binarization;
	img3.convertTo(Binarization,CV_8UC1,255,0);
	imwrite("��ֵ��.jpg",Binarization);

	
	//radon�任
	int length=img3.cols;
	int width = img3.rows;
	Mat img4;
	Mat rotation;
	Mat radon = Mat(256,180,CV_32FC1);//���ڴ洢radon�任����ľ���
	for(int i=0;i<180;i++)
	{
		rotation= getRotationMatrix2D(Point2f(length/2,width/2),i,1.0);
		warpAffine(img3,img4,rotation,Size(length,width));//��ͼ����תi�Ƕ�
		
		for(int j=0;j<256;j++)
		{
			radon.at<float>(j,i) = SUM(img4,j);//�����ת�Ƕ�֮���radonֵ
		}
	}

	normalize(radon,radon,0,1,CV_MINMAX);
	imshow("radon�任",radon);
	Mat Radon;
	radon.convertTo(Radon,CV_8UC1,255,0);
	imwrite("radon�任.jpg",Radon);

	//�ҳ�radon�任�����ĻҶ�ֵ�����Ӧ�ĽǶ����˶�ģ���ǶȻ���
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
	cout<<"�˶�ģ������Ϊ"<<endl<<90-k<<endl;

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
    Mat padded;//��ԭͼ��Ĵ�С��Ϊm*n�Ĵ�С�������λ����0��
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));//�����ǻ�ȡ������mat��һ�����ڴ��dft�任��ʵ����һ�����ڴ���鲿����ʼ��ʱ��ʵ������ͼ�����鲿ȫΪ0
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;//��������ͨ����mat�ںϳ�һ����ͨ����mat�������ںϵ�complexImg����ʵ���������鲿
    merge(planes, 2, complexImg);//dft�任����ΪcomplexImg�����������ͨ����mat������dft�任�Ľ��Ҳ���Ա���������
    dft(complexImg, complexImg);    //��complexImg���²�ֳ�����mat��һ����ʵ����һ�����鲿
    split(complexImg, planes);

    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    //��һ������Ϊ�˼���dft�任��ķ�ֵ���Ա�����ʾ��ֵ�ļ��㹫ʽ����
    magnitude(planes[0], planes[1], planes[0]);//������mat��Ӧλ�����
    mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);

    //�޼�Ƶ�ף����ͼ����л������������Ļ�������Ƶ���ǲ��ԳƵģ����Ҫ�޼�
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