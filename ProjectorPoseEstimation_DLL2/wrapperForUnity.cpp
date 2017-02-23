#include "wrapperForUnity.h"


DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, double trackingtime, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
{
	return static_cast<void *>(new ProjectorEstimation(camWidth, camHeight, proWidth, proHeight, trackingtime, backgroundImgFile, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset));	
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
DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, /*unsigned char* cam_data,*/ 
																int dotsCount, int dots_data[],
																double _initR[], double _initT[], 
																double _dstR[], double _dstT[], 
																double aveError[],
																double _dstR_predict[], double _dstT_predict[],
																double thresh,
																bool isKalman, bool isPredict)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	cv::Mat initR = (cv::Mat_<double>(3,3) << _initR[0], _initR[1], _initR[2], _initR[3], _initR[4], _initR[5], _initR[6], _initR[7], _initR[8] );
	cv::Mat initT = (cv::Mat_<double>(3,1) << _initT[0], _initT[1], _initT[2]);

	//1�t���[����̐���l
	cv::Mat dstR = cv::Mat::eye(3,3,CV_64F);
	cv::Mat dstT = cv::Mat::zeros(3,1,CV_64F);
	cv::Mat error = cv::Mat::zeros(1,1,CV_64F);

	//�\���l
	cv::Mat dstR_predict = cv::Mat::eye(3,3,CV_64F);
	cv::Mat dstT_predict = cv::Mat::zeros(3,1,CV_64F);

	//�J�����摜��Mat�ɕ���
	//cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC3, cam_data); 

	//cv::Mat cam_drawimg = cam_img.clone();
	cv::Mat proj_drawing = pe->proj_img.clone();
	


	while(!pe->detect_proj)//�v���W�F�N�^�摜��̃R�[�i�[���o(�ŏ��̈��̂�)
	{
		//�h�b�g�p�^�[���̃h�b�g���W�̃��[�h
		pe->detect_proj = pe->loadDots(pe->projcorners, proj_drawing);
	}
	
	bool result = false;
	result = pe->findProjectorPose_Corner( pe->proj_img, initR, initT, dstR, dstT, error, dstR_predict, dstT_predict, dotsCount, dots_data, thresh, isKalman, isPredict, /*cam_img,*/ proj_drawing);

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
		aveError[0] = error.at<double>(0, 0);

		//���茋�ʂ��i�[
		_dstR_predict[0] = dstR_predict.at<double>(0,0);
		_dstR_predict[1] = dstR_predict.at<double>(0,1);
		_dstR_predict[2] = dstR_predict.at<double>(0,2);
		_dstR_predict[3] = dstR_predict.at<double>(1,0);
		_dstR_predict[4] = dstR_predict.at<double>(1,1);
		_dstR_predict[5] = dstR_predict.at<double>(1,2);
		_dstR_predict[6] = dstR_predict.at<double>(2,0);
		_dstR_predict[7] = dstR_predict.at<double>(2,1);
		_dstR_predict[8] = dstR_predict.at<double>(2,2);

		_dstT_predict[0] = dstT_predict.at<double>(0, 0);
		_dstT_predict[1] = dstT_predict.at<double>(1, 0);
		_dstT_predict[2] = dstT_predict.at<double>(2, 0);

	}else
	{
	}

	//pe->startTic();
	//�R�[�i�[���o���ʕ\��(5ms)
	cv::Mat /*resize_cam,*/ resize_proj;
	//cv::resize(cam_img, resize_cam, cv::Size(), 0.8, 0.8);
	//cv::imshow("Camera detected corners", resize_cam);

	cv::resize(proj_drawing, resize_proj, cv::Size(), 0.8, 0.8);
	cv::imshow("Projector detected corners", resize_proj);
	//pe->stopTic("show");

	return result;
}

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

	cv::Mat resizeimg, dilatedimg, resultimg;
	cv::Mat element(9,9,CV_8U, cv::Scalar(1));

	cv::bitwise_not(flip_cam_img, flip_cam_img);//�������]
	//�����̃T�C�Y��
	cv::resize(flip_cam_img, resizeimg, cv::Size(), 0.5, 0.5);
	//�c�����������Ƃ�
	cv::dilate(resizeimg, dilatedimg, element, cv::Point(-1,-1), 1);
	//���̑傫���ɖ߂�
	cv::resize(dilatedimg, resultimg, cv::Size(), 2.0, 2.0);
	cv::bitwise_not(resultimg, resultimg);

	pe->CameraMask = resultimg.clone();

	//�ꉞ�ۑ�
	cv::imwrite("CameraMask.png", pe->CameraMask);
}

//�h�b�g�m�F�p
DLLExport void checkDotsArray(void* projectorestimation, unsigned char* cam_data, int dotsCount, int dots_data[])
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//�J�����摜��Mat�ɕ���
	//cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC1, cam_data);  //PGR
	cv::Mat cam_img(1200, 1920, CV_8UC3, cam_data); //pe->camera->height, pe->camera->width���^�C�~���O�I�ɓǂ߂Ȃ��H�H�H�H

	//�h�b�g�z���vector�ɂ���
	std::vector<cv::Point2f> dots;
	for(int i = 0; i < dotsCount*2; i+=2)
	{
		dots.emplace_back(cv::Point2f(dots_data[i], dots_data[i+1]));
	}

	//img�Ƀh�b�g�`��
	for(int i = 0; i < dots.size(); i++)
	{
		//�`��(�J�����摜)
		cv::circle(cam_img, dots[i], 1, cv::Scalar(255, 0, 0), 3); //��
	}

	cv::imshow("dots check", cam_img);
}

