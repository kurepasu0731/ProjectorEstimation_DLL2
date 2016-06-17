#include "projectorPoseEstimation.h"


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
bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat camframe, const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, 
												   int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode, 
												   cv::Mat &draw_camimage, cv::Mat &draw_projimage)
{
	//draw�p(�J����)
	draw_camimage = camframe.clone();

	//�J�����摜��̃R�[�i�[���o
	bool detect_cam = getCorners(camframe, camcorners, camMinDist, camCornerNum, draw_camimage);
	//�v���W�F�N�^�摜��̃R�[�i�[���o
	bool detect_proj = getCorners(projframe, projcorners, projMinDist, projCornerNum, draw_projimage); //projcorners��draw_projimage��ł����̂́A�c�ݏ������ĂȂ�����

	//�R�[�i�[���o�ł�����A�ʒu����J�n
	if(detect_cam && detect_proj)
	{
		// �Ή��_�̘c�ݏ���
		std::vector<cv::Point2f> undistort_imagePoint;
		std::vector<cv::Point2f> undistort_projPoint;
		cv::undistortPoints(camcorners, undistort_imagePoint, camera->cam_K, camera->cam_dist);
		cv::undistortPoints(projcorners, undistort_projPoint, projector->cam_K, projector->cam_dist);
		for(int i=0; i<camcorners.size(); ++i)
		{
			undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera->cam_K.at<double>(0,0) + camera->cam_K.at<double>(0,2);
			undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera->cam_K.at<double>(1,1) + camera->cam_K.at<double>(1,2);
		}
		for(int i=0; i<projcorners.size(); ++i)
		{
			undistort_projPoint[i].x = undistort_projPoint[i].x * projector->cam_K.at<double>(0,0) + projector->cam_K.at<double>(0,2);
			undistort_projPoint[i].y = undistort_projPoint[i].y * projector->cam_K.at<double>(1,1) + projector->cam_K.at<double>(1,2);
		}

		cv::Mat _dstR = cv::Mat::eye(3,3,CV_64F);
		cv::Mat _dstT = cv::Mat::zeros(3,1,CV_64F);

		int result = 0;
		if(mode == 1)
			//result = calcProjectorPose_Corner1(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner1(undistort_imagePoint, projcorners, initialR, initialT, _dstR, _dstT, draw_camimage, draw_projimage);
		else if(mode == 2)
			//result = calcProjectorPose_Corner2(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner2(undistort_imagePoint, projcorners, initialR, initialT, _dstR, _dstT, draw_camimage, draw_projimage);

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);

		if(result > 0) return true;
		else return false;
	}
	else{
		return false;
	}
}

//�v�Z����
int ProjectorEstimation::calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage)
{
		////��]�s�񂩂��]�x�N�g���ɂ���
		//cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
		//Rodrigues(initialR, initRVec);
		//��]�s�񂩂�N�H�[�^�j�I���ɂ���
		cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
		double w, x, y, z;
		transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
		//cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

		int n = 6; //�ϐ��̐�
		int info;
		double level = 1.0; //�����x�N�g���̑傫��

		VectorXd initial(n);
		initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);
		//initial << x, y, z, initTVec.at<double>(0, 0), initTVec.at<double>(1, 0), initTVec.at<double>(2, 0);

		//3�������W����ꂽ�Ή��_�݂̂𒊏o���Ă���LM�@�ɓ����
		std::vector<cv::Point3f> reconstructPoints_valid;
		//�Ή��t�����Ă�J�����摜�_
		std::vector<cv::Point2f> imagePoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * camera->width + image_x;
			//-1�̓v���W�F�N�^���e�̈�O�A-2�̓J�����摜��ł̑Ώە��̃G���A���Ӗ�����
			if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height && reconstructPoints[index].x != -1 && reconstructPoints[index].x != -2)
			{
				//�}�X�N�̈�̃R�[�i�[�_�͏��O
				if(CameraMask.data[index + 0] != 0 && CameraMask.data[index + 1] != 0 && CameraMask.data[index + 2] != 0 )
				{
					reconstructPoints_valid.emplace_back(reconstructPoints[index]);
					imagePoints_valid.emplace_back(imagePoints[i]);
				}
			}
		}

		//�d�S
		//misra2a_functor functor(n, projPoints.size(), projPoints, reconstructPoints_valid, projector.cam_K);
		//NumericalDiff<misra2a_functor> numDiff(functor);
		//LevenbergMarquardt<NumericalDiff<misra2a_functor> > lm(numDiff);

		////�ŋߖT 
		//misra3a_functor functor(n, projPoints.size(), projPoints, reconstructPoints_valid, projector.cam_K);
		//NumericalDiff<misra3a_functor> numDiff(functor);
		//LevenbergMarquardt<NumericalDiff<misra3a_functor> > lm(numDiff);

		///////�����ŋߖT�T���őΉ������߂遫��///////

		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
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

		//�Ή�����3�����_�𐮗񂷂�
		std::vector<cv::Point3f> reconstructPoints_order;
		//�Ή��t�����Ă�J�����摜�_������
		std::vector<cv::Point2f> imagePoints_order;
		
		for(int i = 0; i < projPoints.size(); i++){
			reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
			imagePoints_order.emplace_back(imagePoints_valid[indices[i][0]]);
		}

		///////�����ŋߖT�T���őΉ������߂遪��///////

		misra1a_functor functor(n, projPoints.size(), projPoints, reconstructPoints_order, projector->cam_K, projK_34);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);


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
		for(int i = 0; i < reconstructPoints_order.size(); i++)
		{
			// 2����(�v���W�F�N�^�摜)���ʂ֓��e
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_order[i].x, reconstructPoints_order[i].y, reconstructPoints_order[i].z, 1);
			//4*4�s��ɂ���
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																		  _dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																		  _dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																		  0, 0, 0, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			//�`��(�v���W�F�N�^�摜)
			cv::circle(chessimage, projPoints[i], 5, cv::Scalar(0, 0, 255), 3); //�v���W�F�N�^�͐�
			cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//�J�����͐�
			//�`��(�J�����摜)
			cv::circle(draw_camimage, imagePoints_order[i], 1, cv::Scalar(255, 0, 0), 3); //�Ή������Ă�̂͐�

			double error = sqrt(pow(pt.x - projPoints[i].x, 2) + pow(pt.y - projPoints[i].y, 2));
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
		
		//cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

		int n = 6; //�ϐ��̐�
		int info;
		double level = 1.0; //�����x�N�g���̑傫��

		VectorXd initial(n);
		initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);
		//initial << x, y, z, initTVec.at<double>(0, 0), initTVec.at<double>(1, 0), initTVec.at<double>(2, 0);
		//initial <<
		//	initRVec.at<double>(0, 0) + dR.at<double>(0, 0) * level,
		//	initRVec.at<double>(1, 0) + dR.at<double>(1, 0) * level,
		//	initRVec.at<double>(2, 0) + dR.at<double>(2, 0) * level,
		//	initTVec.at<double>(0, 0) + dt.at<double>(0, 0) * level,
		//	initTVec.at<double>(1, 0) + dt.at<double>(1, 0) * level,
		//	initTVec.at<double>(2, 0) + dt.at<double>(2, 0) * level;

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

		//�Ή�����3�����_�𐮗񂷂�
		std::vector<cv::Point2f> projPoints_order;
		for(int i = 0; i < reconstructPoints_valid.size(); i++){
			projPoints_order.emplace_back(projPoints[indices[i][0]]);
		}

		misra1a_functor functor(n, projPoints_order.size(), projPoints_order, reconstructPoints_valid, projector->cam_K, projK_34);
    
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

		//�����x�N�g���X�V
		//dR = initRVec - dstRVec;
		//dt = initTVec - dstTVec;

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
	//cv::Mat undist_img1;
	////�J�����摜�̘c�ݏ���
	//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
	//�R�[�i�[���o
	//getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

	//�`�F�b�J�p�^�[�����o(�J�����摜�͘c�񂾂܂�)
	bool detect = getCheckerCorners(cameraImageCorners, frame, draw_image);

	//���o�ł�����A�ʒu����J�n
	if(detect)
	{
		// �Ή��_�̘c�ݏ���
		std::vector<cv::Point2f> undistort_imagePoint;
		std::vector<cv::Point2f> undistort_projPoint;
		cv::undistortPoints(cameraImageCorners, undistort_imagePoint, camera->cam_K, camera->cam_dist);
		cv::undistortPoints(projectorImageCorners, undistort_projPoint, projector->cam_K, projector->cam_dist);
		for(int i=0; i<cameraImageCorners.size(); ++i)
		{
			undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera->cam_K.at<double>(0,0) + camera->cam_K.at<double>(0,2);
			undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera->cam_K.at<double>(1,1) + camera->cam_K.at<double>(1,2);
			undistort_projPoint[i].x = undistort_projPoint[i].x * projector->cam_K.at<double>(0,0) + projector->cam_K.at<double>(0,2);
			undistort_projPoint[i].y = undistort_projPoint[i].y * projector->cam_K.at<double>(1,1) + projector->cam_K.at<double>(1,2);
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
	////��]�s�񂩂��]�x�N�g���ɂ���
	//cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
	//Rodrigues(initialR, initRVec);
	//��]�s�񂩂�N�H�[�^�j�I���ɂ���
	cv::Mat initialR_tr = initialR.t();//�֐��̓s����]�u
	double w, x, y, z;
	transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //�ϐ��̐�
	int info;
	double level = 1.0; //�����x�N�g���̑傫��

	VectorXd initial(n);
	initial << x, y, z, initTVec.at<double>(0, 0), initTVec.at<double>(1, 0), initTVec.at<double>(2, 0);
	//initial <<
	//	initRVec.at<double>(0, 0) + dR.at<double>(0, 0) * level,
	//	initRVec.at<double>(1, 0) + dR.at<double>(1, 0) * level,
	//	initRVec.at<double>(2, 0) + dR.at<double>(2, 0) * level,
	//	initTVec.at<double>(0, 0) + dt.at<double>(0, 0) * level,
	//	initTVec.at<double>(1, 0) + dt.at<double>(1, 0) * level,
	//	initTVec.at<double>(2, 0) + dt.at<double>(2, 0) * level;

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
		}
	}

	misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projector->cam_K, projK_34);
	//misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projK_34);
    
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

	//�Ή��_�̗l�q��`��
	//std::vector<cv::Point2f> pt;
	//cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
	for(int i = 0; i < projPoints_valid.size(); i++)
	{
		// 2����(�v���W�F�N�^�摜)���ʂ֓��e
		cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
		//4*4�s��ɂ���
		cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																		_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																		_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																		0, 0, 0, 1);
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

	//�����x�N�g���X�V
	//dR = initRVec - dstRVec;
	//dt = initTVec - dstTVec;

	//std::cout << "-----\ndR: \n" << dR << std::endl;
	//std::cout << "dT: \n" << dt << std::endl;

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



//******�O���ɂ��炷�p********//

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

//�v���W�F�N�^�ʒu����R�A�Ăяo��
DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
																double _initR[], double _initT[], double _dstR[], double _dstT[],
																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode)
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
		result = pe->findProjectorPose(cam_img, initR, initT, dstR, dstT, cam_drawimg, proj_drawing);
	//�R�[�i�[���o�ɂ�鐄��(�v���W�F�N�^�摜�X�Vver 4->1, 5->2)
	else if(mode == 4 || mode == 5)
	{
		//�v���W�F�N�^�摜��Mat�ɕ���
		cv::Mat prj_img(pe->projector->height, pe->projector->width, CV_8UC4, prj_data);
		//�v���W�F�N�^�摜��Unity���Ő������ꂽ�̂ŁA���]�Ƃ�����
		//BGR <-- ARGB �ϊ�
		cv::Mat bgr_img, flip_prj_img;
		std::vector<cv::Mat> bgra;
		cv::split(prj_img, bgra);
		std::swap(bgra[0], bgra[3]);
		std::swap(bgra[1], bgra[2]);
		cv::cvtColor(prj_img, bgr_img, CV_BGRA2BGR);
		//x�����]
		cv::flip(bgr_img, flip_prj_img, 0);

		proj_drawing = flip_prj_img.clone();

		result = pe->findProjectorPose_Corner(cam_img, flip_prj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, mode-3, cam_drawimg, proj_drawing);
	}
	//�R�[�i�[���o�ɂ�鐄��(�v���W�F�N�^�摜�X�V���Ȃ�ver)
	else
	{	proj_drawing = pe->proj_img.clone();
		result = pe->findProjectorPose_Corner(cam_img, pe->proj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, mode, cam_drawimg, proj_drawing);
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
	cv::resize(cam_drawimg, resize_cam, cv::Size(), 0.5, 0.5);
	cv::imshow("Camera detected corners", resize_cam);
	cv::resize(proj_drawing, resize_proj, cv::Size(), 0.5, 0.5);
	cv::imshow("Projector detected corners", resize_proj);

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

	pe->CameraMask = flip_cam_img.clone();

	//�ꉞ�ۑ�
	cv::imwrite("CameraMask.png", pe->CameraMask);
}



