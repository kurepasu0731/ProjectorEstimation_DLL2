#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"
#include "KalmanFilter.h"

#include <opencv2/opencv.hpp>

//PCL
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>


//#define DLLExport __declspec (dllexport)


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;
	cv::Size checkerPattern;

	std::vector<cv::Point2f> projectorImageCorners; //�v���W�F�N�^�摜��̑Ή��_���W(�`�F�b�J�p�^�[���ɂ�鐄��̏ꍇ)
	std::vector<cv::Point2f> cameraImageCorners; //�J�����摜��̑Ή��_���W

	//�J�����摜��̃R�[�i�[�_
	std::vector<cv::Point2f> camcorners;
	//�v���W�F�N�^�摜��̃R�[�i�[�_
	std::vector<cv::Point2f> projcorners;

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

	//�Ή��_�Ƃ��邩�ǂ�����臒l
	//double thresh;

	//�J���}���t�B���^
	Kalmanfilter kf;

	//�R���X�g���N�^
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);
		checkerPattern = cv::Size(_checkerCol, _checkerRow);

		kf.initKalmanfilter(6, 3, 0, 1);

		//�v���W�F�N�^�摜�ǂݍ���,�`��p�摜�쐬
		proj_img = cv::imread(backgroundImgFile);
		proj_undist =  proj_img.clone();
		cv::undistort(proj_img, proj_undist, projector->cam_K, projector->cam_dist);

		//�`�F�b�J�p�^�[���ɂ�鐄��̏ꍇ
		//�v���W�F�N�^�摜��̌�_���W�����߂Ă���
		getProjectorImageCorners(projectorImageCorners, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset);
		
	};

	~ProjectorEstimation(){};

	//�L�����u���[�V�����t�@�C���ǂݍ���
	void loadProCamCalibFile(const std::string& filename);

	//3�����������ʓǂݍ���
	void loadReconstructFile(const std::string& filename);

	//�R�[�i�[���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
	bool findProjectorPose_Corner(const cv::Mat camframe, const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, 
		int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode, cv::Mat &draw_camimage, cv::Mat &draw_projimage);

	//�`�F�b�J�{�[�h���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
	bool findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage);

private:
	//�v�Z����(�v���W�F�N�^�_�̍ŋߖ_��T������)
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh,
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage);

	//�v�Z����(�J�����_(3�����_)�̍ŋߖT��T������)
	int calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage);

	//�R�[�i�[���o
	bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage);

	//�e�Ή��_�̏d�S�ʒu���v�Z
	void calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
									cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve);
	
	//�`�F�b�J�p�^�[���ɂ�鐄��̏ꍇ

	//�v�Z����(R�̎��R�x3)
	int calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//�J�����摜���`�F�b�J�p�^�[�����o����
	bool getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image);

	//�v���W�F�N�^�摜��̌�_���W�����߂�
	void getProjectorImageCorners(std::vector<cv::Point2f>& projPoint, int _row, int _col, int _blockSize, int _x_offset, int _y_offset);

	//��]�s�񁨃N�H�[�^�j�I��
	bool transformRotMatToQuaternion(
		double &qx, double &qy, double &qz, double &qw,
		double m11, double m12, double m13,
		double m21, double m22, double m23,
		double m31, double m32, double m33);


};

/*
extern "C" {
	//ProjectorEstimation�C���X�^���X����
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset, double _thresh); 

	//�p�����[�^�t�@�C���A3���������t�@�C���ǂݍ���
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//�v���W�F�N�^�ʒu����R�A�Ăяo��
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
																	double initR[], double initT[], double dstR[], double dstT[],
																	int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode);
	//�E�B���h�E�j��
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};

	//�J�����摜�p�}�X�N�̍쐬
	DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data);

}
*/
#endif