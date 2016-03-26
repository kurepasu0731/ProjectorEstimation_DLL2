#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"

#include <opencv2/opencv.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>


#define DLLExport __declspec (dllexport)


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;

	std::vector<cv::Point2f> projectorImageCorners; //�v���W�F�N�^�摜��̑Ή��_���W
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
	cv::Mat proj_img, proj_drawimg;


	//�R���X�g���N�^
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);

		//�v���W�F�N�^�摜�ǂݍ���,�`��p�摜�쐬
		proj_img = cv::imread("Assets/Image/bedsidemusic_1280_800.jpg");
		proj_drawimg =  proj_img.clone();
		cv::undistort(proj_img, proj_drawimg, projector->cam_K, projector->cam_dist);
		
	};

	~ProjectorEstimation(){};

	//�L�����u���[�V�����t�@�C���ǂݍ���
	void loadProCamCalibFile(const std::string& filename);

	//3�����������ʓǂݍ���
	void loadReconstructFile(const std::string& filename);

	//�R�[�i�[���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
	bool findProjectorPose_Corner(const cv::Mat& camframe, const cv::Mat projframe, cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, 
		int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode, cv::Mat &draw_camimage, cv::Mat &draw_projimage);

private:
	//�v�Z����(�v���W�F�N�^�_�̍ŋߖ_��T������)
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//�v�Z����(�J�����_(3�����_)�̍ŋߖT��T������)
	int calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//�R�[�i�[���o
	bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage);

	//�e�Ή��_�̏d�S�ʒu���v�Z
	void calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
									cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve);
};

extern "C" {
	//ProjectorEstimation�C���X�^���X����
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight); 

	//�p�����[�^�t�@�C���A3���������t�@�C���ǂݍ���
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//�v���W�F�N�^�ʒu����R�A�Ăяo��
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, 
																	double initR[], double initT[], double dstR[], double dstT[],
																	int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode);
	//�E�B���h�E�j��
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};
}
#endif