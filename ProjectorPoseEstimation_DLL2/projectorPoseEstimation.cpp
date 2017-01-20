#include "projectorPoseEstimation.h"
#include "DebugLogWrapper.h"


//�L�����u���[�V�����t�@�C���ǂݍ���
void ProjectorEstimation::loadProCamCalibFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//�J�����p�����[�^�ǂݍ���
	read(node["cam_K"], camera->cam_K);
	read(node["cam_dist"], camera->cam_dist);
	//�v���W�F�N�^�p�����[�^�ǂݍ���
	read(node["proj_K"], projector->cam_K);
	read(node["proj_dist"], projector->cam_dist);

	read(node["R"], projector->cam_R);
	read(node["T"], projector->cam_T);

	//��Ŏg���v���W�F�N�^�̓����s��
	projK_34 = (cv::Mat_<double>(3, 4) << projector->cam_K.at<double>(0,0),projector->cam_K.at<double>(0,1), projector->cam_K.at<double>(0,2), 0,
						        projector->cam_K.at<double>(1,0), projector->cam_K.at<double>(1,1), projector->cam_K.at<double>(1,2), 0,
								projector->cam_K.at<double>(2,0), projector->cam_K.at<double>(2,1), projector->cam_K.at<double>(2,2), 0);

}

//3���������t�@�C���ǂݍ���
void ProjectorEstimation::loadReconstructFile(const std::string& filename)
{
		//3�����_(�J�������S)LookUp�e�[�u���̃��[�h
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		cv::FileNode node(fs.fs, NULL);

		read(node["points"], reconstructPoints);
}

//�R�[�i�[���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &error,
												   int dotsCount, int dots_data[],
												   //int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, 
												   double thresh, 
												   int mode,
												   bool isKalman, bool isPredict,
												   //double C, int dotsMin, int dotsMax, float resizeScale,
												   cv::Mat &draw_camimage, cv::Mat &draw_projimage)
{
	//�������Ԍv��
	CFileTime cTimeStart_, cTimeEnd_;
	CFileTimeSpan cTimeSpan_;

	//cTimeStart_ = CFileTime::GetCurrentTime();// ���ݎ���

	//startTic();

	bool detect_cam = false;
	if(mode == 4)
	{
		//cv::Mat src;
		//cv::cvtColor(camframe, src, CV_BGR2GRAY);
		////src = camframe.clone(); //PGR
		//detect_cam = getDots(src, camcorners, C, dotsMin, dotsMax, resizeScale, draw_camimage);

		//�h�b�g�z���vector�ɂ���
		camcorners.clear();
		for(int i = 0; i < dotsCount*2; i+=2)
		{
			camcorners.emplace_back(cv::Point2f(dots_data[i], dots_data[i+1]));
		}

		if(camcorners.size() > 0) detect_cam = true;
		else detect_cam = false;
	}
	else
	{
		//�J�����摜��̃R�[�i�[���o
//		detect_cam = getCorners(camframe, camcorners, camMinDist, camCornerNum, draw_camimage);
		//�v���W�F�N�^�摜��̃R�[�i�[���o
		//bool detect_proj = getCorners(projframe, projcorners, projMinDist, projCornerNum, draw_projimage); //projcorners��draw_projimage��ł����̂́A�c�ݏ������ĂȂ�����
	}
	//stopTic("conerDetect");

	//�R�[�i�[���o�ł�����A�ʒu����J�n
	if(detect_cam && detect_proj)
	{
		//startTic();

		// �Ή��_�̘c�ݏ���
//		std::vector<cv::Point2f> undistort_imagePoint;
		//std::vector<cv::Point2f> undistort_projPoint;
		cv::undistortPoints(camcorners, undistort_imagePoint, camera->cam_K, camera->cam_dist);
		//cv::undistortPoints(projcorners, undistort_projPoint, projector->cam_K, projector->cam_dist);
		for(int i=0; i<camcorners.size(); ++i)
		{
			undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera->cam_K.at<double>(0,0) + camera->cam_K.at<double>(0,2);
			undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera->cam_K.at<double>(1,1) + camera->cam_K.at<double>(1,2);
		}
		//for(int i=0; i<projcorners.size(); ++i)
		//{
		//	undistort_projPoint[i].x = undistort_projPoint[i].x * projector->cam_K.at<double>(0,0) + projector->cam_K.at<double>(0,2);
		//	undistort_projPoint[i].y = undistort_projPoint[i].y * projector->cam_K.at<double>(1,1) + projector->cam_K.at<double>(1,2);
		//}

		cv::Mat _dstR = cv::Mat::eye(3,3,CV_64F);
		cv::Mat _dstT = cv::Mat::zeros(3,1,CV_64F);
		cv::Mat  _error = cv::Mat::zeros(1,1,CV_64F);

		int result = 0;
		if(mode == 1 || mode == 4)//�����������@�\���ĂȂ�
			//result = calcProjectorPose_Corner1(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner1(undistort_imagePoint, projcorners, thresh, isKalman, isPredict, initialR, initialT, _dstR, _dstT, _error, draw_camimage, draw_projimage);
		else if(mode == 2)
			//result = calcProjectorPose_Corner2(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner2(undistort_imagePoint, projcorners, initialR, initialT, _dstR, _dstT, draw_camimage, draw_projimage);

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);
		_error.copyTo(error);

		//stopTic("calcConer1");

		//cTimeEnd_ = CFileTime::GetCurrentTime();           // ���ݎ���
		//cTimeSpan_ = cTimeEnd_ - cTimeStart_;
		//debug_log(log);
		//debug_log(std::to_string(cTimeSpan.GetTimeSpan()/10000));
		//std::string timelog = "totalTime: " + std::to_string(cTimeSpan_.GetTimeSpan()/10000);
		//debug_log(timelog);

		if(result > 0) return true;
		else return false;
	}
	else{
		return false;
	}

}

//�v�Z����
int ProjectorEstimation::calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh, bool isKalman, bool isPredict,
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &error, cv::Mat &draw_camimage, cv::Mat &chessimage)
{

		//3�������W����ꂽ�Ή��_�݂̂𒊏o���Ă���LM�@�ɓ����
		std::vector<cv::Point3f> reconstructPoints_valid;
		//�Ή��t�����Ă�J�����摜�_
		std::vector<cv::Point2f> imagePoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * camera->width + image_x;
			//-1�̓v���W�F�N�^���e�̈�O�G���A���Ӗ�����
			if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height &&
				reconstructPoints[index].x != -1.0f && reconstructPoints[index].y != -1.0f && reconstructPoints[index].z != -1.0f)
			{
				//�}�X�N�̈�(�h�������񕔕�)�̃R�[�i�[�_�͏��O
				if(CameraMask.data[index * 3 + 0] != 0 && CameraMask.data[index * 3 + 1] != 0 && CameraMask.data[index * 3 + 2] != 0 )
				{
					reconstructPoints_valid.emplace_back(reconstructPoints[index]);
					imagePoints_valid.emplace_back(imagePoints[i]);
				}
			}
			else
			{
				//�J�����摜�ɕ`��
				cv::circle(draw_camimage, cv::Point(image_x, image_y), 2, cv::Scalar(0, 255, 0), 3); //��
				//std::string stx = "x: ";
				//debug_log(stx);
				//debug_log(std::to_string(reconstructPoints[i].x));
				//std::string sty = "y: ";
				//debug_log(sty);
				//debug_log(std::to_string(reconstructPoints[i].y));
				//std::string stz = "z: ";
				//debug_log(stz);
				//debug_log(std::to_string(reconstructPoints[i].z));
			}
		}

		//std::string deg = "imagePoints:";
		//debug_log(deg);
		//debug_log(std::to_string(imagePoints.size()));
		//std::string deg1 = "projPoints:";
		//debug_log(deg1);
		//debug_log(std::to_string(projPoints.size()));

		//startTic();

		///////�����ŋߖT�T���őΉ������߂遫��///////

		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
		std::vector<cv::Point2d> ppt;
		//4*4�s��ɂ���
		cv::Mat Rt = (cv::Mat_<double>(4, 4) << initialR.at<double>(0,0), initialR.at<double>(0,1), initialR.at<double>(0,2), initialT.at<double>(0,0),
																		initialR.at<double>(1,0), initialR.at<double>(1,1), initialR.at<double>(1,2), initialT.at<double>(1,0),
																		initialR.at<double>(2,0), initialR.at<double>(2,1), initialR.at<double>(2,2), initialT.at<double>(2,0),
																		0, 0, 0, 1);
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2����(�v���W�F�N�^�摜)���ʂ֓��e
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			ppt.emplace_back(pt);
		}
		//�ŋߖT�T�� X:�J�����_�@Y:�v���W�F�N�^�_
		boost::shared_array<double> m_X ( new double [ppt.size()*2] );
		for (int i = 0; i < ppt.size(); i++)
		{
			m_X[i*2 + 0] = ppt[i].x;
			m_X[i*2 + 1] = ppt[i].y;
		}

		flann::Matrix<double> mat_X(m_X.get(), ppt.size(), 2); // Xsize rows and 3 columns

		boost::shared_array<double> m_Y ( new double [projPoints.size()*2] );
		for (int i = 0; i < projPoints.size(); i++)
		{
			m_Y[i*2 + 0] = projPoints[i].x;
			m_Y[i*2 + 1] = projPoints[i].y;
		}
		flann::Matrix<double> mat_Y(m_Y.get(), projPoints.size(), 2); // Ysize rows and 3 columns

		flann::Index< flann::L2<double> > index( mat_X, flann::KDTreeIndexParams() );
		index.buildIndex();
			
		// find closest points
		vector< std::vector<size_t> > indices(projPoints.size());
		vector< std::vector<double> >  dists(projPoints.size());
		//indices[Y�̃C���f�b�N�X][0] = �Ή�����X�̃C���f�b�N�X
		index.knnSearch(mat_Y,
								indices,
								dists,
								1, // k of knn
								flann::SearchParams() );
		///////�����ŋߖT�T���őΉ������߂遪��///////

		//stopTic("Kmeans");

		//�Ή�����3�����_�𐮗񂷂�
		std::vector<cv::Point3f> reconstructPoints_order;
		//�Ή��t�����Ă�J�����摜�_������
		std::vector<cv::Point2f> imagePoints_order;
		//�L���ȃv���W�F�N�^�摜��̑Ή��_
		std::vector<cv::Point2f> projPoints_valid;

		//startTic();

		///////�����Ή��_�̑I������///////

		//**�w�������p�萔**//
		double a = 0.45; //a > 1 -> �ߋ��قǏd���@a < 1 ->�ߋ��قǌy��

		//�Ή��_�Ԃ̋�����臒l�ȏ㗣��Ă�����A�O��l�Ƃ��ď���
		for(int i = 0; i < projPoints.size(); i++)
		{

			//**�ߋ�preframesize�t���[���̉��d����**//
			double distAve = 0;
			//array��30���܂��Ă��Ă��A�����̏d�݂����ς�臒l�ȏ�Ȃ�A�Ή��_�Ƃ��Ēǉ�
			if(preDistsArrays[i].size() == preframesize)
			{
				//�d�݂�����
				for(int j = 0; j < preframesize; j++)
				{
					distAve += ((j + 1) * preDistsArrays[i][j]); //�V�������̂قǏd�݂��d������(<-�������̂ق����悳����)
					//distAve += ((preframesize - j) * preDistsArrays[i][j]); //�V�������̂قǏd�݂��y������
				}
				distAve /= sum;
			}

			////**�w�������@**//
			//double expodist = 0.0;
			//double distance = sqrt(pow(projPoints[i].x - ppt[indices[i][0]].x, 2) + pow(projPoints[i].y - ppt[indices[i][0]].y, 2));
			////t=0�̂Ƃ���dt^ = dt
			//if(preExpoDists[i] == 0.0)
			//{
			//	expodist = distance;
			//}
			//else
			//{
			//	//expodist = a * distance + (1 - a) * preExpoDists[i];//���ʂ̎w�������@
			//	expodist = a * distance + (1/(1 - a)) * preExpoDists[i]; //�ߋ��قǏd������t�w�������@
			//	expodist /= 100; //�Ȃ񂩈Ⴄ�悤�ȋC������
			//}
			//preExpoDists[i] = expodist;

			//std::string logSum = "expo:" + std::to_string(preExpoDists[i]);
			//debug_log(logSum);
			//debug_log(std::to_string(preDistsArrays[i].size()));
			//std::string logAve ="[" + std::to_string(i) + "]: " + std::to_string(distAve);
			//debug_log(logAve);

//			double distance = sqrt(pow(projPoints[i].x - ppt[indices[i][0]].x, 2) + pow(projPoints[i].y - ppt[indices[i][0]].y, 2));
//			if( distance <= thresh + 10 || preDists[i] <= thresh)//���t���[���ł̑Ή��_�ԋ����܂��͑O�t���[���ł̋�����臒l�ȉ��Ȃ�� ->���������Ǝ��Ԏ��ŏd�ݕt���Ƃ�������ǂ��Ȃ肻��
//			if( distance <= thresh) //�P��
//			if(expodist <= thresh)//�w�������@
			if(distAve <= thresh && preDistsArrays[i].size() == preframesize)//���d����
			{
				reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
				imagePoints_order.emplace_back(imagePoints_valid[indices[i][0]]);
				projPoints_valid.emplace_back(projPoints[i]);
			}
			////preDists�̍X�V
			//preDists[i] = distance;
		}

		///////�����Ή��_�̑I������///////

		//stopTic("serect:");

		//�͂����ꂽ�Ή��_���S���`��
		for(int i = 0; i < projPoints.size(); i++)
		{	
			cv::Point pp = cv::Point((int) (projPoints[i].x + 0.5f), (int) (projPoints[i].y + 0.5f));
			cv::Point cp = cv::Point((int) (ppt[indices[i][0]].x + 0.5f), (int) (ppt[indices[i][0]].y + 0.5f));

			//�v���W�F�N�^�摜�̃R�[�i�[�_��`��(�Ή����ĂȂ��̂��S��)
			cv::circle(chessimage, pp, 1, cv::Scalar(0, 0, 255), 3); //�����߂̐�
			cv::putText(chessimage,std::to_string(i), pp, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255));

			//�J�����摜�̃R�[�i�[�_���v���W�F�N�^�摜�֎ˉe�������̂̕`��(�Ή����ĂȂ��̂��S��)
			//�`��(�v���W�F�N�^�摜)
			cv::circle(chessimage, cp, 1, cv::Scalar(255, 0, 0), 3); //�����߂̐�
			cv::putText(chessimage,std::to_string(i), cp, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0));

			//���Ō���
			cv::line(chessimage, pp, cp, cv::Scalar(0, 255, 0), 2);//��
		}

		if(reconstructPoints_order.size() > 20) //10�_�ȏ�c���(�Œ�6�_?)
		{
			//startTic();

			//�p�����[�^�����߂�(�S�_��)			
			cv::Mat _dstR, _dstT;
			//int result = calcParameters(projPoints_valid, reconstructPoints_order, initialR, initialT, _dstR, _dstT);
			//�p�����[�^�����߂�(RANSAC)
			int result = calcParameters_RANSAC(projPoints_valid, reconstructPoints_order, initialR, initialT, 10, thresh, _dstR, _dstT);

			//stopTic("calclate");

			if(isKalman)
			{
				//startTic();

				//--�␳����--//
				cv::Mat translation_measured = (cv::Mat_<double>(3, 1) << _dstT.at<double>(0,0), _dstT.at<double>(1,0), _dstT.at<double>(2,0));
				cv::Mat rotation_measured = (cv::Mat_<double>(3, 3) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2),_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2));
				cv::Mat measurement(6, 1, CV_64F);
				measurement.setTo(cv::Scalar(0));

				kf.fillMeasurements(measurement, translation_measured, rotation_measured);
				// Instantiate estimated translation and rotation
				cv::Mat translation_estimated(3, 1, CV_64F);
				cv::Mat rotation_estimated(3, 3, CV_64F);
				// update the Kalman filter with good measurements
				kf.updateKalmanFilter(measurement, translation_estimated, rotation_estimated);
				cv::Mat _dstT_kf = (cv::Mat_<double>(3, 1) << translation_estimated.at<double>(0, 0), translation_estimated.at<double>(1, 0), translation_estimated.at<double>(2, 0));
				cv::Mat _dstR_kf = (cv::Mat_<double>(3, 3) << rotation_estimated.at<double>(0, 0), rotation_estimated.at<double>(0, 1), rotation_estimated.at<double>(0, 2), rotation_estimated.at<double>(1, 0), rotation_estimated.at<double>(1, 1), rotation_estimated.at<double>(1, 2), rotation_estimated.at<double>(2, 0), rotation_estimated.at<double>(2, 1), rotation_estimated.at<double>(2, 2));
				_dstT_kf.copyTo(_dstT);
				_dstR_kf.copyTo(_dstR);

				//stopTic("Kalman");

			}

			//startTic();

			///////���������\������///////
			// ��]
			glm::mat4 rotation( _dstR.at<double>(0, 0), _dstR.at<double>(1,0), _dstR.at<double>(2,0), 0.0f,
								_dstR.at<double>(0,1), _dstR.at<double>(1,1), _dstR.at<double>(2,1), 0.0f,
								_dstR.at<double>(0,2), _dstR.at<double>(1,2), _dstR.at<double>(2,2), 0.0f,
								0.f, 0.f, 0.f, 1.f);
			timer.stop();
			predict_point->addData(timer.MSec(), cv::Point3f(_dstT.at<double>(0, 0), _dstT.at<double>(1, 0), _dstT.at<double>(2, 0)));
			predict_quat->addData(timer.MSec(), glm::quat_cast(rotation));

			// ���e�\���ʒu
			cv::Point3f predict_point2 = predict_point->calcYValue(timer.MSec()+trackingTime);

			// �x���⏞�����ʒu�p��
			glm::mat4 predictPose = glm::mat4_cast(predict_quat->calcYValue(timer.MSec()+trackingTime));
			predictPose[3][0] = predict_point2.x;
			predictPose[3][1] = predict_point2.y;
			predictPose[3][2] = predict_point2.z;

			// �\���t�B���^�g�p��
			if (isPredict && !firstTime)
			{
				_dstT = (cv::Mat_<double>(3, 1) << predictPose[3][0], predictPose[3][1], predictPose[3][2]);
				_dstR = (cv::Mat_<double>(3, 3) << predictPose[0][0], predictPose[1][0], predictPose[2][0],
												   predictPose[0][1], predictPose[1][1], predictPose[2][1], 
												   predictPose[0][2], predictPose[1][2], predictPose[2][2]);
			}
			else
			{
				firstTime = false;
			}

			//stopTic("predict");

			///////���������\������///////

			////�}�X�N��������
			//for(int y = 0; y < draw_camimage.rows; y++)
			//{
			//	for(int x = 0; x < draw_camimage.cols; x++)
			//	{
			//		if(CameraMask.data[(y * draw_camimage.cols + x) * 3 + 0] == 0 && CameraMask.data[(y * draw_camimage.cols + x) * 3 + 1] == 0 && CameraMask.data[(y * draw_camimage.cols + x) * 3 + 2] == 0)
			//		{
			//				draw_camimage.data[(y * draw_camimage.cols + x) * 3 + 0] = 0; 
			//				draw_camimage.data[(y * draw_camimage.cols + x) * 3 + 1] = 0; 
			//				draw_camimage.data[(y * draw_camimage.cols + x) * 3 + 2] = 0; 
			//		}
			//	}
			//}

			//�Ή��_�̗l�q��`��
			vector<cv::Point2d> projection_P;
			vector<double> errors;
			double aveError = 0; //���ύē��e
			calcReprojectionErrors(projPoints_valid, reconstructPoints_order, _dstR, _dstT, projection_P, errors);
			//calcReprojectionErrors(projPoints_valid, reconstructPoints_order, _dstR, _dstT_kf, projection_P, errors);


			//�L���ȑΉ��_�̕`��(R,t�����)
			for(int i = 0; i < projPoints_valid.size(); i++)
			{
				cv::Point pp = cv::Point((int) (projPoints_valid[i].x + 0.5f), (int) (projPoints_valid[i].y + 0.5f));
				cv::Point cp = cv::Point((int) (projection_P[i].x + 0.5f), (int) (projection_P[i].y + 0.5f));
				cv::Point icp = cv::Point((int) (imagePoints_order[i].x + 0.5f), (int) (imagePoints_order[i].y + 0.5f));

				//�`��(�v���W�F�N�^�摜)
				cv::circle(chessimage,pp, 5, cv::Scalar(0, 0, 255), 3); //�v���W�F�N�^�͐�
				cv::circle(chessimage, cp, 5, cv::Scalar(255, 0, 0), 3);//�J����(�\������)�͐�
				//�`��(�J�����摜)
				cv::circle(draw_camimage, icp, 1, cv::Scalar(255, 0, 0), 3); //�Ή������Ă�̂͐�

				//���Ō���
				cv::line(chessimage, pp, cp, cv::Scalar(255, 0, 255), 4);//�s���N(��)

				aveError += errors[i];
			}

			aveError /= errors.size();
			////�v���W�F�N�^�摜�̑Ή��_�������Ή��t�����Ă��邩�̊���(��)
			//double percent = (projPoints_valid.size() * 100) / projPoints.size();

			//startTic();

			///////�����Ή��_�ԋ����̍X�V����///////
			//�S�_�̑Ή��_�Ƃ̋���(preDists)���X�V
			//4*4�s��ɂ���
			cv::Mat _dstRt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																			_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																			_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																			0, 0, 0, 1);
			for(int i = 0; i < projPoints.size(); i++)
			{
				// 2����(�v���W�F�N�^�摜)���ʂ֓��e
				cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[indices[i][0]].x, reconstructPoints_valid[indices[i][0]].y, reconstructPoints_valid[indices[i][0]].z, 1);
				cv::Mat dst_p = projK_34 * _dstRt * wp;
				cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				double distance = sqrt(pow(projPoints[i].x - pt.x, 2) + pow(projPoints[i].y - pt.y, 2));
				preDists[i] = distance;

				//arrays�ւ̒l�̑��(MAX��30�ŁA��납�炢���E�O�̂������o��)
				if(preDistsArrays[i].size() < preframesize)
				{
					preDistsArrays[i].emplace_back(distance);
				}
				else
				{
					//pop front
					assert(!preDistsArrays[i].empty());
					preDistsArrays[i].erase(preDistsArrays[i].begin());
					//push back
					preDistsArrays[i].emplace_back(distance);
				}
			}
			///////�����Ή��_�ԋ����̍X�V����///////

			//stopTic("serect2");

			_dstR.copyTo(dstR);
			_dstT.copyTo(dstT);
			cv::Mat _error = (cv::Mat_<double>(1, 1) << aveError); 
			_error.copyTo(error);

			//std::string logAve = "aveError: ";
			//std::string logAve2 =std::to_string(aveError);
			//std::string logPer = "valid points: ";
			//std::string logPer2 = std::to_string(percent);
			//debug_log(logAve);
			//debug_log(logAve2);
			//debug_log(logPer);
			//debug_log(logPer2);

			return result;
		}
		else 
		{
			cv::Mat initialRt = (cv::Mat_<double>(4, 4) << initialR.at<double>(0,0), initialR.at<double>(0,1), initialR.at<double>(0,2), initialT.at<double>(0,0),
																			initialR.at<double>(1,0), initialR.at<double>(1,1), initialR.at<double>(1,2), initialT.at<double>(1,0),
																			initialR.at<double>(2,0), initialR.at<double>(2,1), initialR.at<double>(2,2), initialT.at<double>(2,0),
																			0, 0, 0, 1);

			//�������������l�ŎZ�o
			for(int i = 0; i < projPoints.size(); i++)
			{
				// 2����(�v���W�F�N�^�摜)���ʂ֓��e
				cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[indices[i][0]].x, reconstructPoints_valid[indices[i][0]].y, reconstructPoints_valid[indices[i][0]].z, 1);
				cv::Mat dst_p = projK_34 * initialRt * wp;
				cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				double distance = sqrt(pow(projPoints[i].x - pt.x, 2) + pow(projPoints[i].y - pt.y, 2));
				preDists[i] = distance;

				//arrays�ւ̒l�̑��(MAX��30�ŁA��납�炢���E�O�̂������o��)
				if(preDistsArrays[i].size() < preframesize)
				{
					preDistsArrays[i].emplace_back(distance);
				}
				else
				{
					//pop front
					assert(!preDistsArrays[i].empty());
					preDistsArrays[i].erase(preDistsArrays[i].begin());
					//push back
					preDistsArrays[i].emplace_back(distance);
				}
			}

				return 0;
		}
}



//�v�Z����(�ŋߖT�T����3�����_�̕��ɍ��킹��)
int ProjectorEstimation::calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints,
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage)
{
		////��]�s�񂩂��]�x�N�g���ɂ���
		//cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
		//Rodrigues(initialR, initRVec);
		//��]�s�񂩂�N�H�[�^�j�I���ɂ���
		cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
		double w, x, y, z;
		transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		

		int n = 6; //�ϐ��̐�
		int info;

		VectorXd initial(n);
		initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);

		//3�������W����ꂽ�Ή��_�݂̂𒊏o���Ă���LM�@�ɓ����
		std::vector<cv::Point3f> reconstructPoints_valid;
		//�Ή��t�����Ă�J�����摜�_
		std::vector<cv::Point2f> imagePoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * camera->width + image_x;
			if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height && reconstructPoints[index].x != -1)
			{
				reconstructPoints_valid.emplace_back(reconstructPoints[index]);
				imagePoints_valid.emplace_back(imagePoints[i]);
			}
		}

		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
		//std::vector<cv::Point2f> ppt;
		//cv::projectPoints(reconstructPoints_valid, initialR, initTVec, projector.cam_K, cv::Mat(), ppt); 
		std::vector<cv::Point2d> ppt;
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2����(�v���W�F�N�^�摜)���ʂ֓��e
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			//4*4�s��ɂ���
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << initialR.at<double>(0,0), initialR.at<double>(0,1), initialR.at<double>(0,2), initialT.at<double>(0,0),
																		  initialR.at<double>(1,0), initialR.at<double>(1,1), initialR.at<double>(1,2), initialT.at<double>(1,0),
																		  initialR.at<double>(2,0), initialR.at<double>(2,1), initialR.at<double>(2,2), initialT.at<double>(2,0),
																		  0, 0, 0, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			ppt.emplace_back(pt);
		}

		///////�����ŋߖT�T���őΉ������߂遫��///////

		//�ŋߖT�T�� X:�J�����_�@Y:�v���W�F�N�^�_
		boost::shared_array<double> m_X ( new double [ppt.size()*2] );
		for (int i = 0; i < ppt.size(); i++)
		{
			m_X[i*2 + 0] = ppt[i].x;
			m_X[i*2 + 1] = ppt[i].y;
		}
		flann::Matrix<double> mat_X(m_X.get(), ppt.size(), 2); // Xsize rows and 3 columns

		boost::shared_array<double> m_Y ( new double [projPoints.size()*2] );
		for (int i = 0; i < projPoints.size(); i++)
		{
			m_Y[i*2 + 0] = projPoints[i].x;
			m_Y[i*2 + 1] = projPoints[i].y;
		}
		flann::Matrix<double> mat_Y(m_Y.get(), projPoints.size(), 2); // Ysize rows and 3 columns

		flann::Index< flann::L2<double> > index( mat_Y, flann::KDTreeIndexParams() );
		index.buildIndex();
			
		// find closest points
		vector< std::vector<size_t> > indices(reconstructPoints_valid.size());
		vector< std::vector<double> >  dists(reconstructPoints_valid.size());
		//indices[X�̃C���f�b�N�X][0] = �Ή�����Y�̃C���f�b�N�X
		index.knnSearch(mat_X,
								indices,
								dists,
								1, // k of knn
								flann::SearchParams() );

		//�Ή��_�̏d��
		//std::vector<double> weight;
		//�Ή�����3�����_�𐮗񂷂�
		std::vector<cv::Point2f> projPoints_order;
		for(int i = 0; i < reconstructPoints_valid.size(); i++){
			projPoints_order.emplace_back(projPoints[indices[i][0]]);
			////����
			//weight.emplace_back(1.0);
		}

		misra1a_functor functor(n, projPoints_order.size(), projPoints_order, reconstructPoints_valid, projK_34);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

		///////�����ŋߖT�T���őΉ������߂遪��///////

		info = lm.minimize(initial);
    
		std::cout << "�w�K����: " << std::endl;
		std::cout <<
			initial[0] << " " <<
			initial[1] << " " <<
			initial[2] << " " <<
			initial[3] << " " <<
			initial[4] << " " <<
			initial[5]	 << std::endl;

		//�o��
		//cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		//Rodrigues(dstRVec, dstR); //->src.copyTo(data)�g���đ�����Ȃ��ƃ_���@����Ȃ��ā@��]�x�N�g���𖈉񐳋K�����Ȃ��ƃ_��
		//��]
		Quaterniond q(0, initial[0], initial[1], initial[2]);
		q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
		q.normalize ();
		MatrixXd qMat = q.toRotationMatrix();
		cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
		//���i
		cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
		//cv::Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//�ێ��p

		//�Ή��_�̓��e�덷�Z�o
		double aveError = 0;

		//�Ή��_�̗l�q��`��
		//std::vector<cv::Point2f> pt;
		//cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2����(�v���W�F�N�^�摜)���ʂ֓��e
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			//4*4�s��ɂ���
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																		  _dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																		  _dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																		  0, 0, 0, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			//�`��
			cv::circle(chessimage, projPoints_order[i], 5, cv::Scalar(0, 0, 255), 3); //�v���W�F�N�^�͐�
			cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//�J�����͐�
			//�`��(�J�����摜)
			cv::circle(draw_camimage, imagePoints_valid[i], 1, cv::Scalar(255, 0, 0), 3); //�Ή������Ă�̂͐�

			double error = sqrt(pow(pt.x - projPoints_order[i].x, 2) + pow(pt.y - projPoints_order[i].y, 2));
			aveError += error;
		}

		////�d�S���`��
		//cv::Point2f imageWorldPointAve;
		//cv::Point2f projAve;
		//calcAveragePoint(reconstructPoints_valid, projPoints, dstRVec, dstTVec,imageWorldPointAve, projAve);
		//cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//�v���W�F�N�^�͐�
		//cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//�J�����͐�

		std::cout << "reprojection error ave : " << (double)(aveError / projPoints.size()) << std::endl;

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);

		std::cout << "info: " << info << std::endl;
		return info;
}





//�R�[�i�[���o
bool ProjectorEstimation::getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage){
	cv::Mat gray_img;
	//�c�ݏ���
	//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
	//�O���[�X�P�[��
	cv::cvtColor(frame, gray_img, CV_BGR2GRAY);

	//�R�[�i�[���o
	//int num = 500;
	cv::goodFeaturesToTrack(gray_img, corners, num, 0.01, minDistance);

	//�����x��
	cv::cornerSubPix(gray_img, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

	//�`��
	for(int i = 0; i < corners.size(); i++)
	{
		cv::circle(drawimage, corners[i], 1, cv::Scalar(0, 0, 255), 3);
	}

	//�R�[�i�[���o���ł������ǂ���
	if(corners.size() > 0)	return true;
	else	return false;
}


//�e�Ή��_�̏d�S�ʒu���v�Z
void ProjectorEstimation::calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
								cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve)
{
	//�e�Ή��_�̃v���W�F�N�^�摜��ł̏d�S�����߂�
	//(1)proj_p_
	float sum_px = 0, sum_py = 0, px = 0, py = 0;
	for(int i = 0; i < projPoints.size(); i++)
	{
		sum_px += projPoints[i].x;
		sum_py += projPoints[i].y;
	}
	px = sum_px / projPoints.size();
	py = sum_py / projPoints.size();

	projAve.x = px;
	projAve.y = py;

	//(2)worldPoints_
	// 2����(�v���W�F�N�^�摜)���ʂ֓��e
	std::vector<cv::Point2f> pt;
	cv::projectPoints(imageWorldPoints, R, t, projector->cam_K, cv::Mat(), pt); 
	float sum_wx = 0, sum_wy = 0, wx = 0, wy = 0;
	for(int i = 0; i < pt.size(); i++)
	{
		sum_wx += pt[i].x;
		sum_wy += pt[i].y;
	}
	wx = sum_wx / pt.size();
	wy = sum_wy / pt.size();

	imageAve.x = wx;
	imageAve.y = wy;
}


//****�`�F�b�J�p�^�[���ɂ�鐄��̏ꍇ****

//�`�F�b�J�{�[�h���o�ɂ��v���W�F�N�^�ʒu�p���𐄒�
bool ProjectorEstimation::findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage){

	//�`�F�b�J�p�^�[�����o(�J�����摜�͘c�񂾂܂�)
	bool detect = getCheckerCorners(cameraImageCorners, frame, draw_image);

	//���o�ł�����A�ʒu����J�n
	if(detect)
	{
		// �Ή��_�̘c�ݏ���
		std::vector<cv::Point2f> undistort_imagePoint;
		//std::vector<cv::Point2f> undistort_projPoint;
		cv::undistortPoints(cameraImageCorners, undistort_imagePoint, camera->cam_K, camera->cam_dist);
		//cv::undistortPoints(projectorImageCorners, undistort_projPoint, projector->cam_K, projector->cam_dist);
		for(int i=0; i<cameraImageCorners.size(); ++i)
		{
			undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera->cam_K.at<double>(0,0) + camera->cam_K.at<double>(0,2);
			undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera->cam_K.at<double>(1,1) + camera->cam_K.at<double>(1,2);
			//undistort_projPoint[i].x = undistort_projPoint[i].x * projector->cam_K.at<double>(0,0) + projector->cam_K.at<double>(0,2);
			//undistort_projPoint[i].y = undistort_projPoint[i].y * projector->cam_K.at<double>(1,1) + projector->cam_K.at<double>(1,2);
		}
		cv::Mat _dstR = cv::Mat::eye(3,3,CV_64F);
		cv::Mat _dstT = cv::Mat::zeros(3,1,CV_64F);
			
		int result = calcProjectorPose(undistort_imagePoint, projectorImageCorners, initialR, initialT, _dstR, _dstT, chessimage);

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);

		if(result > 0) return true;

		else return false;
	}
	else{
		return false;
	}
}

//�v�Z����(R�̎��R�x3)
int ProjectorEstimation::calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage)
{
	//��]�s�񂩂�N�H�[�^�j�I���ɂ���
	cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
	double w, x, y, z;
	transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //�ϐ��̐�
	int info;

	VectorXd initial(n);
	initial << x, y, z, initTVec.at<double>(0, 0), initTVec.at<double>(1, 0), initTVec.at<double>(2, 0);

	//�Ή��_�̏d��(�_�~�[)
	//std::vector<double> weight;

	//3�������W����ꂽ�Ή��_�݂̂𒊏o���Ă���LM�@�ɓ����
	std::vector<cv::Point3f> reconstructPoints_valid;
	std::vector<cv::Point2f> projPoints_valid;
	for(int i = 0; i < imagePoints.size(); i++)
	{
		int image_x = (int)(imagePoints[i].x+0.5);
		int image_y = (int)(imagePoints[i].y+0.5);
		int index = image_y * camera->width + image_x;
		if(reconstructPoints[index].x != -1)
		{
			reconstructPoints_valid.emplace_back(reconstructPoints[index]);
			projPoints_valid.emplace_back(projPoints[i]);
			//weight.emplace_back(1.0);
		}
	}

	misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projK_34);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
	info = lm.minimize(initial); //info=2���������Ă��� ���X5
    
	//std::cout << "�w�K����: " << std::endl;
	//std::cout <<
	//	initial[0] << " " <<
	//	initial[1] << " " <<
	//	initial[2] << " " <<
	//	initial[3] << " " <<
	//	initial[4] << " " <<
	//	initial[5]	 << std::endl;

	//�o��
	//��]
	Quaterniond q(0, initial[0], initial[1], initial[2]);
	q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
	q.normalize ();
	MatrixXd qMat = q.toRotationMatrix();
	cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
	//���i
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);

	//�Ή��_�̗l�q��`��
	//std::vector<cv::Point2f> pt;
	//cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
	//4*4�s��ɂ���
	cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																	_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																	_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																	0, 0, 0, 1);
	for(int i = 0; i < projPoints_valid.size(); i++)
	{
		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
		cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
		cv::Mat dst_p = projK_34 * Rt * wp;
		cv::Point2f pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
		//�`��
		cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //�v���W�F�N�^�͐�
		cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//�J�����͐�
	}
	////�d�S���`��
	//cv::Point2f imageWorldPointAve;
	//cv::Point2f projAve;
	//calcAveragePoint(reconstructPoints_valid, projPoints_valid, dstRVec, dstTVec,imageWorldPointAve, projAve);
	//cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//�v���W�F�N�^�͐�
	//cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//�J�����͐�


	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);
	std::cout << "info: " << info << std::endl;
	return info;
}

//�J�����摜���`�F�b�J�p�^�[�����o����
bool ProjectorEstimation::getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image)
{
	//��_���o
	bool detect = cv::findChessboardCorners(image, checkerPattern, imagePoint);

	//���o�_�̕`��
	image.copyTo(draw_image);
	if(detect)
	{
		//�T�u�s�N�Z�����x
		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		cv::cornerSubPix( gray, imagePoint, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.001 ) );

		cv::drawChessboardCorners(draw_image, checkerPattern, imagePoint, true);
	}else
	{
		cv::drawChessboardCorners(draw_image, checkerPattern, imagePoint, false);
	}

	return detect;
}


//�v���W�F�N�^�摜��̌�_���W�����߂�
void ProjectorEstimation::getProjectorImageCorners(std::vector<cv::Point2f>& projPoint, int _row, int _col, int _blockSize, int _x_offset, int _y_offset)
{
	for (int y = 0; y < _row; y++)
	{
		for(int x = 0; x < _col; x++)
		{
			projPoint.push_back(cv::Point2f(_x_offset + x * _blockSize, _y_offset + y * _blockSize));
		}
	}
}

///////////////////////////////////////////////
// ��]�s�񁨃N�H�[�^�j�I���ϊ�
//
// qx, qy, qz, qw : �N�H�[�^�j�I�������i�o�́j
// m11-m33 : ��]�s�񐬕�
//
// �����ӁF
// �s�񐬕���DirectX�`���i�s���������̌����j�ł�
// OpenGL�`���i����������̌����j�̏ꍇ��
// �]�u�����l�����ĉ������B

bool ProjectorEstimation::transformRotMatToQuaternion(
    double &qx, double &qy, double &qz, double &qw,
    double m11, double m12, double m13,
    double m21, double m22, double m23,
    double m31, double m32, double m33
) {
    // �ő听��������
    double elem[ 4 ]; // 0:x, 1:y, 2:z, 3:w
    elem[ 0 ] = m11 - m22 - m33 + 1.0f;
    elem[ 1 ] = -m11 + m22 - m33 + 1.0f;
    elem[ 2 ] = -m11 - m22 + m33 + 1.0f;
    elem[ 3 ] = m11 + m22 + m33 + 1.0f;

    unsigned biggestIndex = 0;
    for ( int i = 1; i < 4; i++ ) {
        if ( elem[i] > elem[biggestIndex] )
            biggestIndex = i;
    }

    if ( elem[biggestIndex] < 0.0f )
        return false; // �����̍s��ɊԈႢ����I

    // �ő�v�f�̒l���Z�o
    double *q[4] = {&qx, &qy, &qz, &qw};
    double v = sqrtf( elem[biggestIndex] ) * 0.5f;
    *q[biggestIndex] = v;
    double mult = 0.25f / v;

    switch ( biggestIndex ) {
    case 0: // x
        *q[1] = (m12 + m21) * mult;
        *q[2] = (m31 + m13) * mult;
        *q[3] = (m23 - m32) * mult;
        break;
    case 1: // y
        *q[0] = (m12 + m21) * mult;
        *q[2] = (m23 + m32) * mult;
        *q[3] = (m31 - m13) * mult;
        break;
    case 2: // z
        *q[0] = (m31 + m13) * mult;
        *q[1] = (m23 + m32) * mult;
        *q[3] = (m12 - m21) * mult;
    break;
    case 3: // w
        *q[0] = (m23 - m32) * mult;
        *q[1] = (m31 - m13) * mult;
        *q[2] = (m12 - m21) * mult;
        break;
    }

    return true;
}


//�����_����num�_�𒊏o
void ProjectorEstimation::get_random_points(int num, vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, vector<cv::Point2f>& calib_p, vector<cv::Point3f>& calib_P){
	int i=0;
	//������
	calib_p.clear();
	calib_P.clear();

	//srand(time(NULL));    /* �����̏����� */ 
	std::random_device rnd;//rand() < 32767 std::random_device < 0xffffffff=4294967295
	cv::Vector<int> exists;
	while(i < num){
		int maxValue = (int)src_p.size();
		int v = rand() % maxValue;
		//int v = rnd() % maxValue;
		bool e2=false;
		for(int s=0; s<i; s++){
			if(exists[s] == v) e2 = true; 
		}
		if(!e2){
			exists.push_back(v);
			calib_P.push_back(src_P[v]);
			calib_p.push_back(src_p[v]);
			i++;
		}
	}
}

//�Ή��_����R��T�̎Z�o
int ProjectorEstimation::calcParameters(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT){
	//��]�s�񂩂�N�H�[�^�j�I���ɂ���
	cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
	double w, x, y, z;
	transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), 
																initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), 
																initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
	int n = 6; //�ϐ��̐�
	int info;

	VectorXd initial(n);
	initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);

	misra1a_functor functor(n, src_p.size(), src_p, src_P, projK_34);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

	//�œK��
	info = lm.minimize(initial);//->�������d��
    
	//�o��
	//��]
	Quaterniond q(0, initial[0], initial[1], initial[2]);
	q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
	q.normalize ();
	MatrixXd qMat = q.toRotationMatrix();
	cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
	//���i
	//--�\���Ȃ�--//
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);

	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);

	return info;
}

//Ceres Solver Version(�}�C�i�X30ms���炢�ɂȂ����I�I�I)
int ProjectorEstimation::calcParameters_Ceres(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
{
	////��]�s�񂩂�N�H�[�^�j�I���ɂ���
	//cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
	//double w, x, y, z;
	//transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), 
	//															initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), 
	//															initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
	//double camera[6] = {x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0) };
	
	//��]�s�񂩂�I�C���[�ɂ���
	cv::Mat initial_euler = kf.rot2euler(initialR); 
	double camera[6] = {initial_euler.at<double>(0, 0), initial_euler.at<double>(1, 0), initial_euler.at<double>(2, 0), initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0) };


	ceres::Problem problem;
	for(int i = 0; i < src_p.size(); i++)
	{

		cv::Point2d p(src_p[i].x, src_p[i].y);
		cv::Point3d P(src_P[i].x, src_P[i].y, src_P[i].z);
		CostFunction* cost = new AutoDiffCostFunction<ErrorFunction, 2, 6>(new ErrorFunction(p, P, projK_34));
		problem.AddResidualBlock(cost, NULL, camera);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//Quaterniond q(0, camera[0], camera[1], camera[2]);
	//q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
	//q.normalize ();
	//MatrixXd qMat = q.toRotationMatrix();
	//cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));

	//�o��
	//��]
	cv::Mat dst_euler = (cv::Mat_<double>(3, 1) << camera[0], camera[1], camera[2]);
	cv::Mat _dstR = kf.euler2rot(dst_euler);

	//���i
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << camera[3], camera[4], camera[5]);

	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);

	return summary.IsSolutionUsable();
}


//3�����_�̃v���W�F�N�^�摜�ւ̎ˉe�ƍē��e�덷�̌v�Z
void ProjectorEstimation::calcReprojectionErrors(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat R, cv::Mat T, vector<cv::Point2d>& projection_P, vector<double>& errors){
	//�Ή��_�̗l�q��`��
	cv::Mat Rt = (cv::Mat_<double>(4, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), T.at<double>(0,0),
																	R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), T.at<double>(1,0),
																	R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), T.at<double>(2,0),
																	0, 0, 0, 1);
	for(int i = 0; i < src_P.size(); i++)
	{
		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
		cv::Mat wp = (cv::Mat_<double>(4, 1) << src_P[i].x, src_P[i].y, src_P[i].z, 1);
		//4*4�s��ɂ���
		//--�\���Ȃ�--//
		cv::Mat dst_p = projK_34 * Rt * wp;
		cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
		//�Ή��_�̍ē��e�덷�Z�o
		double reprojectError = sqrt(pow(pt.x - src_p[i].x, 2) + pow(pt.y - src_p[i].y, 2));

		projection_P.emplace_back(pt);
		errors.emplace_back(reprojectError);
	}
}

//�Ή��_����R��T�̎Z�o(RANSAC)
int ProjectorEstimation::calcParameters_RANSAC(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT,int num, float thresh, cv::Mat& dstR, cv::Mat& dstT)
{
		//inlier�̊���
		double maxpercentage = 0.0;
		//�ő�X�R�A
		int maxscore = 0;
		//inlier�Ή��_
		std::vector<cv::Point3f> inlier_P;
		std::vector<cv::Point2f> inlier_p;

		//�J��Ԃ���
		int iterate = 0;

		//�����_���ɑI��ł���_
		std::vector<cv::Point3f> random_P;
		std::vector<cv::Point2f> random_p;

		//�ē��e�덷
		vector<cv::Point2d> projection_P;
		vector<double> errors;

		while(iterate < 10)
		{
			//�N���A
			random_P.clear();
			random_p.clear();

			//1. �����_����num�_�I��
			get_random_points(num, src_p, src_P, random_p, random_P);

			//2. num�_�Ńp�����[�^�����߂�			
			cv::Mat preR, preT;//���p�����[�^
			int result = calcParameters_Ceres(random_p, random_P, initialR, initialT, preR, preT);
			//int result = calcParameters(random_p, random_P, initialR, initialT, preR, preT);

			//debug_log(std::to_string(result));

			if(result > 0)
			{
				projection_P.clear();
				errors.clear();

				//3. �S�_�ōē��e�덷�����߂�
				calcReprojectionErrors(src_p, src_P, preR, preT, projection_P, errors);

				//4. �ē��e�덷��臒l�ȉ����������̂̊�����99���ȉ���������A���s
				int score = 0;
				for(int i = 0; i < errors.size(); i++)
				{
					if(errors[i] <= thresh)
					{
						score++;
					}
				}

				if(score >= maxscore)
				{
					//�N���A
					inlier_P.clear();
					inlier_p.clear();

					for(int i = 0; i < errors.size(); i++)
					{
						if(errors[i] <= thresh)
						{
							inlier_p.emplace_back(src_p[i]);
							inlier_P.emplace_back(src_P[i]);
						}
					}
					maxpercentage = score * 100 / src_p.size();
					maxscore = score;
				}

				if(maxpercentage >= 90)
				{
					//debug_log(std::to_string(maxpercentage) + "%");
					//debug_log("iter:" + std::to_string(iterate));
					break;
				}
				iterate++;
			}
		}
		//debug_log(std::to_string(maxpercentage) + "%");
		//debug_log("iter:" + std::to_string(iterate));

		//5. inlier�ōēx�p�����[�^�����߂�
		cv::Mat final_R, final_T;
		//debug_log("final");
		int result = calcParameters_Ceres(inlier_p, inlier_P, initialR, initialT, final_R, final_T); //->�����߂����Ⴉ����(21~26ms)
		//int result = calcParameters(inlier_p, inlier_P, initialR, initialT, final_R, final_T); //->�����߂����Ⴉ����(21~26ms)

		//�Ή��_�̗l�q��`��
		projection_P.clear();
		errors.clear();
		double aveError = 0; //���ύē��e

		calcReprojectionErrors(inlier_p, inlier_P, final_R, final_T, projection_P, errors);

		for(int i = 0; i < inlier_p.size(); i++)
		{
			aveError += errors[i];
		}
		aveError /= errors.size();

		final_R.copyTo(dstR);
		final_T.copyTo(dstT);

		return result;
}

//�������Ԍv���p�EDebugLog�\���p
void ProjectorEstimation::stopTic(std::string label)
{
		cTimeEnd = CFileTime::GetCurrentTime();           // ���ݎ���
		cTimeSpan = cTimeEnd - cTimeStart;
		//debug_log(log);
		//debug_log(std::to_string(cTimeSpan.GetTimeSpan()/10000));
		std::string timelog = label +": " + std::to_string(cTimeSpan.GetTimeSpan()/10000);
		debug_log(timelog);
}

//**�����_���h�b�g�}�[�J�[�p**//	
//csv�t�@�C������~�̍��W��ǂݍ���
bool ProjectorEstimation::loadDots(std::vector<cv::Point2f> &corners, cv::Mat &drawimage)
{
	string filename = "Calibration/dots.csv";

    //�t�@�C���̓ǂݍ���
    ifstream ifs(filename);
    if(!ifs){
        return false;
    }

    //csv�t�@�C����1�s���ǂݍ���
    string str;
    while(getline(ifs,str)){
        string token;
        istringstream stream(str);

		//x���W
		getline(stream,token,',');
		int x = std::stoi(token);
		//y���W
		getline(stream,token,',');
		int y = std::stoi(token);

		corners.emplace_back(cv::Point2f(x, y));

	}
	return true;
}
//�h�b�g���o
bool ProjectorEstimation::getDots(cv::Mat &src, std::vector<cv::Point2f> &dots, double C, int dots_thresh_min, int dots_thresh_max, float resizeScale, cv::Mat &drawimage)
{
	dots.clear();
	//���T�C�Y
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(), resizeScale, resizeScale);
	//�K���I臒l����
	cv::adaptiveThreshold(resized, resized, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, C);
	//�c������
	cv::dilate(resized, resized, cv::Mat());
	//������ɖ߂�
	cv::Mat ptsImg = cv::Mat::zeros( camera->height, camera->width, CV_8UC1); //������\���p
	cv::resize(resized, ptsImg, cv::Size(), 1/resizeScale, 1/resizeScale);
	//cv::Mat ptsImgColor; 
	//cv::cvtColor(ptsImg, ptsImgColor, CV_GRAY2BGR);

	cv::Point sum, min, max, p;
	int cnt;
	for (int i = 0; i < ptsImg.rows; i++) {
		for (int j = 0; j < ptsImg.cols; j++) {
			if (ptsImg.at<uchar>(i, j)) {
				sum = cv::Point(0, 0); cnt = 0; min = cv::Point(j, i); max = cv::Point(j, i);
				calCoG_dot_v0(ptsImg, sum, cnt, min, max, cv::Point(j, i));
				if (cnt>dots_thresh_min && max.x - min.x < dots_thresh_max && max.y - min.y < dots_thresh_max) {
					dots.push_back(cv::Point(sum.x / cnt, sum.y / cnt));
					//dots.push_back(cv::Point((int)((float)(sum.x / cnt) / resizeScale + 0.5), (int)((float)(sum.y / cnt) / resizeScale + 0.5)));

				}
			}
		}
	}


	//cv::rectangle(ptsImg, cv::Point(CamWidth / 4, CamHeight / 4), cv::Point(CamWidth * 3 / 4, CamHeight * 3 / 4), cv::Scalar(255, 0, 0), 5, 4);
	// OpenGL�p�ɗ\��RGB�p�̃f�[�^�쐬
	//cv::rectangle(ptsImg, cv::Point(CamWidth / 4, CamHeight / 4), cv::Point(CamWidth * 3 / 4, CamHeight * 3 / 4), cv::Scalar(0, 0, 255), 5, 4);
	std::vector<cv::Point2f>::iterator it = dots.begin();
	
	//bool k = (dots.size()==projcorners.size());
	bool k = (dots.size() >= 20);
	//for (int i=0; it != dots.end(); i++,++it) {
	//	if (i && i%MarkersWidth == 0) {
	//		if ((*it).y <= (*(it - 1)).y) k = false;
	//	} else {
	//		if (i && (*it).x <= (*(it - 1)).x) k = false;
	//	}
	//}
	//cv::Scalar co = k ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
	// OpenGL�p�ɗ\��RGB�p�̃f�[�^�쐬
	cv::Scalar color = k ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
	for (it = dots.begin(); it != dots.end(); ++it) {
		cv::circle(drawimage, *it, 3, color, 2);
		//cv::circle(ptsImgColor, *it, 3, color, 2);
	}
	
	//cv::imshow("detected dots", ptsImgColor); 

	//if (k) {
	//	it = dots.begin();
	//	for (int i = 0; it != dots.end(); ++it) {
	//		marker_u[i / MarkersWidth][i%MarkersWidth] = *it;
	//		marker_s[i / MarkersWidth][i%MarkersWidth] = true;
	//		i++;
	//	}
	//	
	//	flag = 1;
	//	std::cout << "init complete!" << std::endl;
	//	
	//}

	return k;
}

void ProjectorEstimation::calCoG_dot_v0(cv::Mat &src, cv::Point& sum, int& cnt, cv::Point& min, cv::Point& max, cv::Point p)
{
	if (src.at<uchar>(p)) {
		sum += p; cnt++;
		src.at<uchar>(p) = 0;
		if (p.x<min.x) min.x = p.x;
		if (p.x>max.x) max.x = p.x;
		if (p.y<min.y) min.y = p.y;
		if (p.y>max.y) max.y = p.y;

		if (p.x - 1 >= 0) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x-1, p.y));
		if (p.x + 1 < camera->width) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x + 1, p.y));
		if (p.y - 1 >= 0) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x, p.y - 1));
		if (p.y + 1 < camera->height) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x, p.y + 1));
	}
}

