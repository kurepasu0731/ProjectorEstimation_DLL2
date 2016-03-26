#ifndef WEBCAMERA_H
#define WEBCAMERA_H

#include <opencv2\opencv.hpp>

class WebCamera
{
public:
	//解像度
	int width;
	int height;

	// カメラパラメータ
	cv::Mat cam_K;					// 内部パラメータ行列
	cv::Mat cam_dist;				// レンズ歪み
	cv::Mat cam_R;					// 回転ベクトル
	cv::Mat cam_T;					// 平行移動ベクトル
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