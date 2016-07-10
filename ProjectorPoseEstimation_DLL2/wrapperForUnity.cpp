#include "wrapperForUnity.h"

DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
{
	return static_cast<void *>(new ProjectorEstimation(camWidth, camHeight, proWidth, proHeight, backgroundImgFile, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset));	
}

//�p�����[�^�t�@�C���A3���������t�@�C���ǂݍ���
DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[])
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	pe->loadReconstructFile("Calibration/reconstructPoints_camera.xml");
	pe->loadProCamCalibFile("Calibration/calibration.xml");

	initR[0] = pe->projector->cam_R.at<double>(0, 0);
	initR[1] = pe->projector->cam_R.at<double>(0, 1);
	initR[2] = pe->projector->cam_R.at<double>(0, 2);
	initR[3] = pe->projector->cam_R.at<double>(1, 0);
	initR[4] = pe->projector->cam_R.at<double>(1, 1);
	initR[5] = pe->projector->cam_R.at<double>(1, 2);
	initR[6] = pe->projector->cam_R.at<double>(2, 0);
	initR[7] = pe->projector->cam_R.at<double>(2, 1);
	initR[8] = pe->projector->cam_R.at<double>(2, 2);

	initT[0] = pe->projector->cam_T.at<double>(0, 0);
	initT[1] = pe->projector->cam_T.at<double>(1, 0);
	initT[2] = pe->projector->cam_T.at<double>(2, 0);

}


//�v���W�F�N�^�ʒu����R�A�Ăяo��(�v���W�F�N�^�摜�X�V�Ȃ�)
DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, 
																double _initR[], double _initT[], double _dstR[], double _dstT[],
																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//�J�����摜��Mat�ɕ���
	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC3, cam_data);

	cv::Mat initR = (cv::Mat_<double>(3,3) << _initR[0], _initR[1], _initR[2], _initR[3], _initR[4], _initR[5], _initR[6], _initR[7], _initR[8] );
	cv::Mat initT = (cv::Mat_<double>(3,1) << _initT[0], _initT[1], _initT[2]);

	//1�t���[����̐���l
	cv::Mat dstR = cv::Mat::eye(3,3,CV_64F);
	cv::Mat dstT = cv::Mat::zeros(3,1,CV_64F);

	cv::Mat cam_drawimg = cam_img.clone();
	cv::Mat proj_drawing;

	bool result = false;

	//�ʒu���胁�\�b�h�Ăяo��
	if(mode == 3)//�`�F�b�J�p�^�[�����o�ɂ�鐄��
	{	proj_drawing = pe->proj_img.clone();
		result = pe->findProjectorPose(cam_img, initR, initT, dstR, dstT, cam_drawimg, proj_drawing);
	}
	//�R�[�i�[���o�ɂ�鐄��(�v���W�F�N�^�摜�X�V���Ȃ�ver)
	else
	{	proj_drawing = pe->proj_img.clone();
		result = pe->findProjectorPose_Corner(cam_img, pe->proj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, thresh, mode, cam_drawimg, proj_drawing);
	}

	if(result)
	{
		//���茋�ʂ��i�[
		_dstR[0] = dstR.at<double>(0,0);
		_dstR[1] = dstR.at<double>(0,1);
		_dstR[2] = dstR.at<double>(0,2);
		_dstR[3] = dstR.at<double>(1,0);
		_dstR[4] = dstR.at<double>(1,1);
		_dstR[5] = dstR.at<double>(1,2);
		_dstR[6] = dstR.at<double>(2,0);
		_dstR[7] = dstR.at<double>(2,1);
		_dstR[8] = dstR.at<double>(2,2);

		_dstT[0] = dstT.at<double>(0, 0);
		_dstT[1] = dstT.at<double>(1, 0);
		_dstT[2] = dstT.at<double>(2, 0);
	}else
	{
	}

	//�R�[�i�[���o���ʕ\��
	cv::Mat resize_cam, resize_proj;
	//�}�X�N��������
	for(int y = 0; y < cam_drawimg.rows; y++)
	{
		for(int x = 0; x < cam_drawimg.cols; x++)
		{
			if(pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 0] == 0 && pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 1] == 0 && pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 2] == 0)
			{
					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 0] = 0; 
					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 1] = 0; 
					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 2] = 0; 
			}
		}
	}
	cv::resize(cam_drawimg, resize_cam, cv::Size(), 0.5, 0.5);

	cv::imshow("Camera detected corners", resize_cam);
	cv::resize(proj_drawing, resize_proj, cv::Size(), 0.5, 0.5);
	cv::imshow("Projector detected corners", resize_proj);

	return result;
}

//�v���W�F�N�^�ʒu����R�A�Ăяo��(�v���W�F�N�^�摜�X�V����)

//DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
//																double _initR[], double _initT[], double _dstR[], double _dstT[],
//																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode)
//{
//	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);
//
//	//�J�����摜��Mat�ɕ���
//	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC3, cam_data);
//
//	cv::Mat initR = (cv::Mat_<double>(3,3) << _initR[0], _initR[1], _initR[2], _initR[3], _initR[4], _initR[5], _initR[6], _initR[7], _initR[8] );
//	cv::Mat initT = (cv::Mat_<double>(3,1) << _initT[0], _initT[1], _initT[2]);
//
//	//1�t���[����̐���l
//	cv::Mat dstR = cv::Mat::eye(3,3,CV_64F);
//	cv::Mat dstT = cv::Mat::zeros(3,1,CV_64F);
//
//	cv::Mat cam_drawimg = cam_img.clone();
//	cv::Mat proj_drawing;
//
//	bool result = false;
//
//	//�ʒu���胁�\�b�h�Ăяo��
//	if(mode == 3)//�`�F�b�J�p�^�[�����o�ɂ�鐄��
//	{	proj_drawing = pe->proj_img.clone();
//		result = pe->findProjectorPose(cam_img, initR, initT, dstR, dstT, cam_drawimg, proj_drawing);
//	}
//	//�R�[�i�[���o�ɂ�鐄��(�v���W�F�N�^�摜�X�Vver 4->1, 5->2)
//	else if(mode == 4 || mode == 5)
//	{
//		//�v���W�F�N�^�摜��Mat�ɕ���
//		cv::Mat prj_img(pe->projector->height, pe->projector->width, CV_8UC4, prj_data);
//		//�v���W�F�N�^�摜��Unity���Ő������ꂽ�̂ŁA���]�Ƃ�����
//		//BGR <-- ARGB �ϊ�
//		cv::Mat bgr_img, flip_prj_img;
//		std::vector<cv::Mat> bgra;
//		cv::split(prj_img, bgra);
//		std::swap(bgra[0], bgra[3]);
//		std::swap(bgra[1], bgra[2]);
//		cv::cvtColor(prj_img, bgr_img, CV_BGRA2BGR);
//		//x�����]
//		cv::flip(bgr_img, flip_prj_img, 0);
//
//		proj_drawing = flip_prj_img.clone();
//
//		result = pe->findProjectorPose_Corner(cam_img, flip_prj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, thresh, mode-3, cam_drawimg, proj_drawing);
//	}
//	//�R�[�i�[���o�ɂ�鐄��(�v���W�F�N�^�摜�X�V���Ȃ�ver)
//	else
//	{	proj_drawing = pe->proj_img.clone();
//		result = pe->findProjectorPose_Corner(cam_img, pe->proj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, thresh, mode, cam_drawimg, proj_drawing);
//	}
//
//	if(result)
//	{
//		//���茋�ʂ��i�[
//		_dstR[0] = dstR.at<double>(0,0);
//		_dstR[1] = dstR.at<double>(0,1);
//		_dstR[2] = dstR.at<double>(0,2);
//		_dstR[3] = dstR.at<double>(1,0);
//		_dstR[4] = dstR.at<double>(1,1);
//		_dstR[5] = dstR.at<double>(1,2);
//		_dstR[6] = dstR.at<double>(2,0);
//		_dstR[7] = dstR.at<double>(2,1);
//		_dstR[8] = dstR.at<double>(2,2);
//
//		_dstT[0] = dstT.at<double>(0, 0);
//		_dstT[1] = dstT.at<double>(1, 0);
//		_dstT[2] = dstT.at<double>(2, 0);
//	}else
//	{
//	}
//
//	//�R�[�i�[���o���ʕ\��
//	cv::Mat resize_cam, resize_proj;
//	//�}�X�N��������
//	for(int y = 0; y < cam_drawimg.rows; y++)
//	{
//		for(int x = 0; x < cam_drawimg.cols; x++)
//		{
//			if(pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 0] == 0 && pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 1] == 0 && pe->CameraMask.data[(y * cam_drawimg.cols + x) * 3 + 2] == 0)
//			{
//					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 0] = 0; 
//					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 1] = 0; 
//					cam_drawimg.data[(y * cam_drawimg.cols + x) * 3 + 2] = 0; 
//			}
//		}
//	}
//	cv::resize(cam_drawimg, resize_cam, cv::Size(), 0.5, 0.5);
//
//	cv::imshow("Camera detected corners", resize_cam);
//	cv::resize(proj_drawing, resize_proj, cv::Size(), 0.5, 0.5);
//	cv::imshow("Projector detected corners", resize_proj);
//
//	return result;
//}

//�J�����摜�p�}�X�N�̍쐬
DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//�J�����摜��Mat�ɕ���
	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC4, cam_data);
	//�v���W�F�N�^�摜��Unity���Ő������ꂽ�̂ŁA���]�Ƃ�����
	//BGR <-- ARGB �ϊ�
	cv::Mat bgr_img, flip_cam_img;
	std::vector<cv::Mat> bgra;
	cv::split(cam_img, bgra);
	std::swap(bgra[0], bgra[3]);
	std::swap(bgra[1], bgra[2]);
	cv::cvtColor(cam_img, bgr_img, CV_BGRA2BGR);
	//x�����]
	cv::flip(bgr_img, flip_cam_img, 0);

	pe->CameraMask = flip_cam_img.clone();

	//�ꉞ�ۑ�
	cv::imwrite("CameraMask.png", pe->CameraMask);
}
