#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#pragma once

#include <opencv2\opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

class Kalmanfilter
{
public:
	cv::KalmanFilter KF;
	int nStates; //状態ベクトルの次元数
	int nMeasurements; //観測ベクトルの次元数
	int nInputs; //the number of action control
	double dt; //time between measurements(1/FPS)

	Kalmanfilter()
	{
	}

	~Kalmanfilter(){};

	//初期化
	void initKalmanfilter(int _nStates, int _nMeasurements, int _nInputs, int _dt);

	//観測値の成形
	void fillMeasurements( cv::Mat &measurements, const cv::Mat &translation_measured, const cv::Mat &rotation_measured);


	//観測値を登録し、更新、予測値の取得
	void updateKalmanFilter(cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated );

	//回転行列→オイラー
	cv::Mat rot2euler(const cv::Mat & rotationMatrix);

	//オイラー→回転行列
	cv::Mat euler2rot(const cv::Mat & euler);



};

#endif