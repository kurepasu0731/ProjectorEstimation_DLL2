#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"
#include "KalmanFilter.h"
#include "ErrorFunction.h"


#include <opencv2/opencv.hpp>
#include <random>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>

#include <atltime.h>

#include<fstream>
#include<iostream>
#include<string>
#include<sstream> //�����X�g���[��

#include "LSM/LSMPoint3f.h"
#include "LSM/LSMQuatd.h"
#include "myTimer.h"


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;
	cv::Size checkerPattern;

	//�J�����摜��̃R�[�i�[�_
	std::vector<cv::Point2f> camcorners;
	//�v���W�F�N�^�摜��̃R�[�i�[�_
	std::vector<cv::Point2f> projcorners;

	//�c�ݏ�����̃J�����摜�̃R�[�i�[�_
	std::vector<cv::Point2f> undistort_imagePoint;


	//3�����_(�J�������S)LookUp�e�[�u��
	//** index = �J������f(����n�܂�)
	//** int image_x = i % CAMERA_WIDTH;
	//** int image_y = (int)(i / CAMERA_WIDTH);
	//** Point3f = �J������f��3�������W(�v������Ă��Ȃ��ꍇ��(-1, -1, -1))
	std::vector<cv::Point3f> reconstructPoints;

	//�v���W�F�N�^�摜
	cv::Mat proj_img, proj_undist;

	//3 * 4�`���̂̃v���W�F�N�^�����s��
	cv::Mat projK_34;

	//�J�����摜(�c�݂Ȃ�)�̃}�X�N
	cv::Mat CameraMask;

	//�J���}���t�B���^
	Kalmanfilter kf;

	//1�t���[���O�̑Ή��_�ԋ���
	std::vector<double> preDists;

	//�w�������@�p1�t���[���O��dt^
	std::vector<double> preExpoDists;

	//�ߋ�preframesize�t���[�����̑Ή��_�ԋ���
	std::vector<std::vector<double>> preDistsArrays;
	//�ǂ̂��炢�ߋ��̏��܂ł݂邩�̃t���[����
	int preframesize;
	int sum;

	//�����\���֘A//
	double trackingTime;		// �g���b�L���O�ɂ����鎞��
	std::unique_ptr<LSMPoint3f> predict_point;	// �ʒu�̍ŏ����@
	std::unique_ptr<LSMQuatd> predict_quat;		// �p���̍ŏ����@
	bool firstTime;								// 1��ڂ��ǂ���
	MyTimer timer;				// �N�����Ԃ���̎��Ԍv��

	//�������Ԍv��
	CFileTime cTimeStart, cTimeEnd;
	CFileTimeSpan cTimeSpan;

	//--�t���O�֌W--//
	bool detect_proj; //�v���W�F�N�^�摜�̃R�[�i�[�_�����o�������ǂ���

	//�R���X�g���N�^
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight, double trackingtime, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);
		checkerPattern = cv::Size(_checkerCol, _checkerRow);

		kf.initKalmanfilter(18, 6, 0, 0.03);//�����x

		//�v���W�F�N�^�摜�ǂݍ���,�`��p�摜�쐬
		proj_img = cv::imread(backgroundImgFile);
		proj_undist =  proj_img.clone();
		cv::undistort(proj_img, proj_undist, projector->cam_K, projector->cam_dist);


		////1�t���[���O�̑Ή��_�ԋ����̏�����
		//for(int i = 0; i < _checkerCol*_checkerRow; i++)
		//{
		//	preDists.emplace_back(0.0);
		//	preExpoDists.emplace_back(0.0);
		//}

		//std::vector<double> array;
		//array.clear();
		//for(int i = 0; i < _checkerCol*_checkerRow; i++)
		//{
		//	preDistsArrays.emplace_back(array);
		//}

		preframesize = 20; 
		//sum�v�Z
		sum = 0;
		for(int i = 1; i <= preframesize; i++)
		{
			sum += i;
		}

		//�R�[�i�[���o�̏ꍇ
		//TODO:�v���W�F�N�^�摜��̃R�[�i�[�_�����߂Ă���
		detect_proj = false;

		//�����\���֘A//
		predict_point = std::unique_ptr<LSMPoint3f> (new LSMPoint3f(LSM));
		predict_quat = std::unique_ptr<LSMQuatd> (new LSMQuatd(LSM));
		predict_point->setForgetFactor(0.6);	// �Y�p�W��
		predict_quat->setForgetFactor(0.6);	// �Y�p�W��
		firstTime = true;
		trackingTime = trackingtime;//�������ԁ{�V�X�e���x��
		timer = MyTimer();
		timer.start();
	};

	~ProjectorEstimation(){};

	//�L�����u���[�V�����t�@�C���ǂݍ���
	void loadProCamCalibFile(const std::string& filename);

	//3�����������ʓǂݍ���
	void loadReconstructFile(const std::string& filename);

	//�R�[�i�[���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
	bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat projframe, 
														cv::Mat initialR, cv::Mat initialT, 
														cv::Mat &dstR, cv::Mat &dstT, 
														cv::Mat &error,
														cv::Mat &dstR_predict, cv::Mat &dstT_predict,
														int dotsCount, int dots_data[],
														double thresh, 
														bool isKalman, bool isPredict,
													   /*cv::Mat &draw_camimage,*/ cv::Mat &draw_projimage);

	//�������Ԍv���EDebugLog�\���p
	void startTic()
	{
		cTimeStart = CFileTime::GetCurrentTime();// ���ݎ���
	}

	//�������Ԍv���p�EDebugLog�\���p
	//�����񂪒�������ƕ���������
	void stopTic(std::string label);

	//csv�t�@�C������~�̍��W��ǂݍ���
	bool ProjectorEstimation::loadDots(std::vector<cv::Point2f> &corners, cv::Mat &drawimage);


private:
	//�v�Z����(�v���W�F�N�^�_�̍ŋߖ_��T������)
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh, bool isKalman, bool isPredict,
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &error, cv::Mat& dstR_predict, cv::Mat& dstT_predict,
												/*cv::Mat &draw_camimage,*/ cv::Mat &chessimage);
	//�����_����num�_�𒊏o
	void get_random_points(int num, vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, vector<cv::Point2f>& calib_p, vector<cv::Point3f>& calib_P);

	//�Ή��_����R��T�̎Z�o
	int calcParameters(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT);
	//Ceres Solver Version
	int ProjectorEstimation::calcParameters_Ceres(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT);

	//3�����_�̃v���W�F�N�^�摜�ւ̎ˉe�ƍē��e�덷�̌v�Z
	void calcReprojectionErrors(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat R, cv::Mat T, vector<cv::Point2d>& projection_P, vector<double>& errors);

	//�Ή��_����R��T�̎Z�o(RANSAC)
	int calcParameters_RANSAC(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, int num, float thresh, cv::Mat& dstR, cv::Mat& dstT);

	//��]�s�񁨃N�H�[�^�j�I��
	bool transformRotMatToQuaternion(
		double &qx, double &qy, double &qz, double &qw,
		double m11, double m12, double m13,
		double m21, double m22, double m23,
		double m31, double m32, double m33);

};
#endif