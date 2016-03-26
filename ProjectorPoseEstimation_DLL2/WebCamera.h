#ifndef WEBCAMERA_H
#define WEBCAMERA_H

#include <opencv2\opencv.hpp>

class WebCamera
{
public:
	//�𑜓x
	int width;
	int height;

	// �J�����p�����[�^
	cv::Mat cam_K;					// �����p�����[�^�s��
	cv::Mat cam_dist;				// �����Y�c��
	cv::Mat cam_R;					// ��]�x�N�g��
	cv::Mat cam_T;					// ���s�ړ��x�N�g��
	WebCamera(){
		cam_K = cv::Mat::eye(3, 3, CV_64F);
		cam_dist = cv::Mat::zeros(1, 5, CV_64F);
		cam_R = cv::Mat::eye(3, 3, CV_64F);
		cam_T = cv::Mat::zeros(3, 1, CV_64F);

		width = 0;
		height = 0;
	};

	WebCamera(int _width, int _height)
	{
		cam_K = cv::Mat::eye(3, 3, CV_64F);
		cam_dist = cv::Mat::zeros(1, 5, CV_64F);
		cam_R = cv::Mat::eye(3, 3, CV_64F);
		cam_T = cv::Mat::zeros(3, 1, CV_64F);

		width = _width;
		height = _height;
	};

	~WebCamera(){};
};
#endif