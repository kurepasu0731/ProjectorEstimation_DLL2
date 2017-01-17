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
	int nStates; //��ԃx�N�g���̎�����
	int nMeasurements; //�ϑ��x�N�g���̎�����
	int nInputs; //the number of action control
	double dt; //time between measurements(1/FPS)

	Kalmanfilter()
	{
	}

	~Kalmanfilter(){};

	//������
	void initKalmanfilter(int _nStates, int _nMeasurements, int _nInputs, int _dt);

	//�ϑ��l�̐��`
	void fillMeasurements( cv::Mat &measurements, const cv::Mat &translation_measured, const cv::Mat &rotation_measured);


	//�ϑ��l��o�^���A�X�V�A�\���l�̎擾
	void updateKalmanFilter(cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated );

	//��]�s�񁨃I�C���[
	cv::Mat rot2euler(const cv::Mat & rotationMatrix);

	//�I�C���[����]�s��
	cv::Mat euler2rot(const cv::Mat & euler);



};

#endif