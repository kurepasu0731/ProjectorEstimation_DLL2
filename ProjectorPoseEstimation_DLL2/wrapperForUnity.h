#ifndef WRAPPERFORUNITY_H
#define WRAPPERFORUNITY_H

#pragma once

#include "projectorPoseEstimation.h"

#define DLLExport __declspec (dllexport)

extern "C" {
	//ProjectorEstimation�C���X�^���X����
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset); 

	//�p�����[�^�t�@�C���A3���������t�@�C���ǂݍ���
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//�v���W�F�N�^�ʒu����R�A�Ăяo��(�v���W�F�N�^�摜�X�V�Ȃ�)
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, 
																	double initR[], double initT[], double dstR[], double dstT[],
																	int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode, bool isKalman);

	//�v���W�F�N�^�ʒu����R�A�Ăяo��(�v���W�F�N�^�摜�X�V����)

	//DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
	//																double initR[], double initT[], double dstR[], double dstT[],
	//																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode);

	//�E�B���h�E�j��
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};

	//�J�����摜�p�}�X�N�̍쐬
	DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data);

    
}
#endif