#include "projectorPoseEstimation.h"


//キャリブレーションファイル読み込み
void ProjectorEstimation::loadProCamCalibFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//カメラパラメータ読み込み
	read(node["cam_K"], camera->cam_K);
	read(node["cam_dist"], camera->cam_dist);
	//プロジェクタパラメータ読み込み
	read(node["proj_K"], projector->cam_K);
	read(node["proj_dist"], projector->cam_dist);

	read(node["R"], projector->cam_R);
	read(node["T"], projector->cam_T);
}

//3次元復元ファイル読み込み
void ProjectorEstimation::loadReconstructFile(const std::string& filename)
{
		//3次元点(カメラ中心)LookUpテーブルのロード
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		cv::FileNode node(fs.fs, NULL);

		read(node["points"], reconstructPoints);
}

//コーナー検出によるプロジェクタ位置姿勢を推定
bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat& camframe, const cv::Mat projframe, 
																		cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, 
																		int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode, 
																		cv::Mat &draw_camimage, cv::Mat &draw_projimage)
{
	//draw用(カメラ)
	draw_camimage = camframe.clone();

	//カメラ画像上のコーナー検出
	bool detect_cam = getCorners(camframe, camcorners, camMinDist, camCornerNum, draw_camimage);
	//プロジェクタ画像上のコーナー検出
	bool detect_proj = getCorners(projframe, projcorners, projMinDist, projCornerNum, draw_projimage); //projcornersがdraw_projimage上でずれるのは、歪み除去してないから

	//コーナー検出できたら、位置推定開始
	if(detect_cam && detect_proj)
	{
		// 対応点の歪み除去
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

		int result = 0;
		if(mode == 1)
			//result = calcProjectorPose_Corner1(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner1(undistort_imagePoint, projcorners, initialR, initialT, dstR, dstT, draw_projimage);

		else if(mode == 2)
			//result = calcProjectorPose_Corner2(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner2(undistort_imagePoint, projcorners, initialR, initialT, dstR, dstT, draw_projimage);

		if(result > 0) return true;
		else return false;
	}
	else{
		return false;
	}
}

//計算部分
int ProjectorEstimation::calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
																		cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &drawimage)
{
	//回転行列から回転ベクトルにする
	cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
	Rodrigues(initialR, initRVec);
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //変数の数
	int info;

	VectorXd initial(n);
	initial <<
		initRVec.at<double>(0, 0),
		initRVec.at<double>(1, 0),
		initRVec.at<double>(2, 0),
		initTVec.at<double>(0, 0),
		initTVec.at<double>(1, 0),
		initTVec.at<double>(2, 0);

	//3次元座標が取れた対応点のみを抽出してからLM法に入れる
	std::vector<cv::Point3f> reconstructPoints_valid;
	for(int i = 0; i < imagePoints.size(); i++)
	{
		int image_x = (int)(imagePoints[i].x+0.5);
		int image_y = (int)(imagePoints[i].y+0.5);
		int index = image_y * camera->width + image_x;
		if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height && reconstructPoints[index].x != -1)
		{
			reconstructPoints_valid.emplace_back(reconstructPoints[index]);
		}
	}

	//↓↓最近傍探索で対応を求める↓↓//

	// 2次元(プロジェクタ画像)平面へ投影
	std::vector<cv::Point2f> ppt;
	cv::projectPoints(reconstructPoints_valid, initialR, initTVec, projector->cam_K, cv::Mat(), ppt); 

	//最近傍探索 X:カメラ点　Y:プロジェクタ点
	boost::shared_array<float> m_X ( new float [ppt.size()*2] );
	for (int i = 0; i < ppt.size(); i++)
	{
		m_X[i*2 + 0] = ppt[i].x;
		m_X[i*2 + 1] = ppt[i].y;
	}

	flann::Matrix<float> mat_X(m_X.get(), ppt.size(), 2); // Xsize rows and 3 columns

	boost::shared_array<float> m_Y ( new float [projPoints.size()*2] );
	for (int i = 0; i < projPoints.size(); i++)
	{
		m_Y[i*2 + 0] = projPoints[i].x;
		m_Y[i*2 + 1] = projPoints[i].y;
	}
	flann::Matrix<float> mat_Y(m_Y.get(), projPoints.size(), 2); // Ysize rows and 3 columns

	flann::Index< flann::L2<float> > index( mat_X, flann::KDTreeIndexParams() );
	index.buildIndex();
			
	// find closest points
	vector< std::vector<size_t> > indices(projPoints.size());
	vector< std::vector<float> >  dists(projPoints.size());
	//indices[Yのインデックス][0] = 対応するXのインデックス
	index.knnSearch(mat_Y,
							indices,
							dists,
							1, // k of knn
							flann::SearchParams() );

	//対応順に3次元点を整列する
	std::vector<cv::Point3f> reconstructPoints_order;
	for(int i = 0; i < projPoints.size(); i++){
		reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
	}

	misra1a_functor functor(n, projPoints.size(), projPoints, reconstructPoints_order, projector->cam_K);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

	//↑↑最近傍探索で対応を求める↑↑//

	info = lm.minimize(initial);
    
	std::cout << "学習結果: " << std::endl;
	std::cout <<
		initial[0] << " " <<
		initial[1] << " " <<
		initial[2] << " " <<
		initial[3] << " " <<
		initial[4] << " " <<
		initial[5]	 << std::endl;

	//出力
	cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
	cv::Rodrigues(dstRVec, dstR);
	dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
	cv::Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

	//対応点の様子を描画
	std::vector<cv::Point2f> pt;
	cv::projectPoints(reconstructPoints_order, dstRVec, dstTVec, projector->cam_K, cv::Mat(), pt); 
	for(int i = 0; i < projPoints.size(); i++)
	{
		cv::circle(drawimage, projPoints[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
	}
	for(int i = 0; i < pt.size(); i++)
	{
		cv::circle(drawimage, pt[i], 5, cv::Scalar(255, 0, 0), 3);//カメラは青
	}

	//重心も描画
	cv::Point2f imageWorldPointAve;
	cv::Point2f projAve;
	calcAveragePoint(reconstructPoints_valid, projPoints, dstRVec, dstTVec,imageWorldPointAve, projAve);
	cv::circle(drawimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
	cv::circle(drawimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

	double aveError = 0;

	//対応点の投影誤差算出
	for(int i = 0; i < projPoints.size(); i++)
	{
		double error = sqrt(pow(pt[i].x - projPoints[i].x, 2) + pow(pt[i].y - projPoints[i].y, 2));
		aveError += error;
		//std::cout << "reprojection error[" << i << "]: " << error << std::endl;

	}
		std::cout << "reprojection error ave : " << (double)(aveError / projPoints.size()) << std::endl;

	//動きベクトル更新
	//dR = initRVec - dstRVec;
	//dt = initTVec - dstTVec;

	std::cout << "info: " << info << std::endl;
	return info;
}



//計算部分(最近傍探索を3次元点の方に合わせる)
int ProjectorEstimation::calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
																		cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &drawimage)
{
	//回転行列から回転ベクトルにする
	cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
	Rodrigues(initialR, initRVec);
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //変数の数
	int info;

	VectorXd initial(n);
	initial <<
		initRVec.at<double>(0, 0),
		initRVec.at<double>(1, 0),
		initRVec.at<double>(2, 0),
		initTVec.at<double>(0, 0),
		initTVec.at<double>(1, 0),
		initTVec.at<double>(2, 0);

	//3次元座標が取れた対応点のみを抽出してからLM法に入れる
	std::vector<cv::Point3f> reconstructPoints_valid;
	for(int i = 0; i < imagePoints.size(); i++)
	{
		int image_x = (int)(imagePoints[i].x+0.5);
		int image_y = (int)(imagePoints[i].y+0.5);
		int index = image_y * camera->width + image_x;
		if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height && reconstructPoints[index].x != -1)
		{
			reconstructPoints_valid.emplace_back(reconstructPoints[index]);
		}
	}

	//↓↓最近傍探索で対応を求める↓↓//

	// 2次元(プロジェクタ画像)平面へ投影
	std::vector<cv::Point2f> ppt;
	cv::projectPoints(reconstructPoints_valid, initialR, initTVec, projector->cam_K, cv::Mat(), ppt); 

	//最近傍探索 X:カメラ点　Y:プロジェクタ点
	boost::shared_array<float> m_X ( new float [ppt.size()*2] );
	for (int i = 0; i < ppt.size(); i++)
	{
		m_X[i*2 + 0] = ppt[i].x;
		m_X[i*2 + 1] = ppt[i].y;
	}
	flann::Matrix<float> mat_X(m_X.get(), ppt.size(), 2); // Xsize rows and 3 columns

	boost::shared_array<float> m_Y ( new float [projPoints.size()*2] );
	for (int i = 0; i < projPoints.size(); i++)
	{
		m_Y[i*2 + 0] = projPoints[i].x;
		m_Y[i*2 + 1] = projPoints[i].y;
	}
	flann::Matrix<float> mat_Y(m_Y.get(), projPoints.size(), 2); // Ysize rows and 3 columns

	flann::Index< flann::L2<float> > index( mat_Y, flann::KDTreeIndexParams() );
	index.buildIndex();
			
	// find closest points
	vector< std::vector<size_t> > indices(reconstructPoints_valid.size());
	vector< std::vector<float> >  dists(reconstructPoints_valid.size());
	//indices[Yのインデックス][0] = 対応するXのインデックス
	index.knnSearch(mat_X,
							indices,
							dists,
							1, // k of knn
							flann::SearchParams() );

	//対応順に3次元点を整列する
	std::vector<cv::Point2f> projPoints_order;
	for(int i = 0; i < reconstructPoints_valid.size(); i++){
		projPoints_order.emplace_back(projPoints[indices[i][0]]);
	}

	misra1a_functor functor(n, projPoints_order.size(), projPoints_order, reconstructPoints_valid, projector->cam_K);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

	//↑↑最近傍探索で対応を求める↑↑//

	info = lm.minimize(initial);
    
	std::cout << "学習結果: " << std::endl;
	std::cout <<
		initial[0] << " " <<
		initial[1] << " " <<
		initial[2] << " " <<
		initial[3] << " " <<
		initial[4] << " " <<
		initial[5]	 << std::endl;

	//出力
	cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
	cv::Rodrigues(dstRVec, dstR);
	dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
	cv::Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

	//対応点の様子を描画
	std::vector<cv::Point2f> pt;
	cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector->cam_K, cv::Mat(), pt); 
	for(int i = 0; i < reconstructPoints_valid.size(); i++)
	{
		cv::circle(drawimage, projPoints_order[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
	}
	for(int i = 0; i < pt.size(); i++)
	{
		cv::circle(drawimage, pt[i], 5, cv::Scalar(255, 0, 0), 3);//カメラは青
	}

	//重心も描画
	cv::Point2f imageWorldPointAve;
	cv::Point2f projAve;
	calcAveragePoint(reconstructPoints_valid, projPoints_order, dstRVec, dstTVec,imageWorldPointAve, projAve);
	cv::circle(drawimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
	cv::circle(drawimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

	double aveError = 0;

	//対応点の投影誤差算出
	for(int i = 0; i < reconstructPoints_valid.size(); i++)
	{
		aveError += sqrt(pow(pt[i].x - projPoints_order[i].x, 2) + pow(pt[i].y - projPoints_order[i].y, 2));
	}
		std::cout << "reprojection error ave : " << (double)(aveError / reconstructPoints_valid.size()) << std::endl;

	//動きベクトル更新
	//dR = initRVec - dstRVec;
	//dt = initTVec - dstTVec;

	std::cout << "info: " << info << std::endl;
	return info;
}

//コーナー検出
bool ProjectorEstimation::getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage){
	cv::Mat gray_img;
	//歪み除去
	//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
	//グレースケール
	cv::cvtColor(frame, gray_img, CV_BGR2GRAY);

	//コーナー検出
	//int num = 500;
	cv::goodFeaturesToTrack(gray_img, corners, num, 0.001, minDistance);

	//描画
	for(int i = 0; i < corners.size(); i++)
	{
		cv::circle(drawimage, corners[i], 1, cv::Scalar(0, 0, 255), 3);
	}

	//コーナー検出ができたかどうか
	if(corners.size() > 0)	return true;
	else	return false;

}


//各対応点の重心位置を計算
void ProjectorEstimation::calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
								cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve)
{
	//各対応点のプロジェクタ画像上での重心を求める
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
	// 2次元(プロジェクタ画像)平面へ投影
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


//****チェッカパターンによる推定の場合****

//チェッカボード検出によるプロジェクタ位置姿勢を推定
bool ProjectorEstimation::findProjectorPose(cv::Mat frame, cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage){
	//cv::Mat undist_img1;
	////カメラ画像の歪み除去
	//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
	//コーナー検出
	//getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

	//チェッカパターン検出(カメラ画像は歪んだまま)
	bool detect = getCheckerCorners(cameraImageCorners, frame, draw_image);

	//検出できたら、位置推定開始
	if(detect)
	{
		// 対応点の歪み除去
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

		//int result = calcProjectorPose(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, chessimage);
		int result = calcProjectorPose(undistort_imagePoint, projectorImageCorners, initialR, initialT, dstR, dstT, chessimage);
		if(result > 0) return true;
		else return false;
	}
	else{
		return false;
	}
}

//計算部分(Rの自由度3)
int ProjectorEstimation::calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage)
{
	//回転行列から回転ベクトルにする
	cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
	Rodrigues(initialR, initRVec);
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //変数の数
	int info;
	double level = 1.0; //動きベクトルの大きさ

	VectorXd initial(n);
	initial <<
		initRVec.at<double>(0, 0),
		initRVec.at<double>(1, 0),
		initRVec.at<double>(2, 0),
		initTVec.at<double>(0, 0),
		initTVec.at<double>(1, 0),
		initTVec.at<double>(2, 0);

	//3次元座標が取れた対応点のみを抽出してからLM法に入れる
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

	misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projector->cam_K);
	//misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projK_34);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
	info = lm.minimize(initial); //info=2がかえってくる 時々5
    
	//std::cout << "学習結果: " << std::endl;
	//std::cout <<
	//	initial[0] << " " <<
	//	initial[1] << " " <<
	//	initial[2] << " " <<
	//	initial[3] << " " <<
	//	initial[4] << " " <<
	//	initial[5]	 << std::endl;

	//出力
	cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
	Rodrigues(dstRVec, dstR);
	dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
	cv::Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

	//対応点の様子を描画
	std::vector<cv::Point2f> pt;
	cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector->cam_K, cv::Mat(), pt); 
	for(int i = 0; i < projPoints_valid.size(); i++)
	{
		cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
		cv::circle(chessimage, pt[i], 5, cv::Scalar(255, 0, 0), 3);//カメラは青
	}
	//重心も描画
	cv::Point2f imageWorldPointAve;
	cv::Point2f projAve;
	calcAveragePoint(reconstructPoints_valid, projPoints_valid, dstRVec, dstTVec,imageWorldPointAve, projAve);
	cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
	cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

	//動きベクトル更新
	//dR = initRVec - dstRVec;
	//dt = initTVec - dstTVec;

	//std::cout << "-----\ndR: \n" << dR << std::endl;
	//std::cout << "dT: \n" << dt << std::endl;


	std::cout << "info: " << info << std::endl;
	return info;
}

//カメラ画像をチェッカパターン検出する
bool ProjectorEstimation::getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image)
{
	//交点検出
	bool detect = cv::findChessboardCorners(image, checkerPattern, imagePoint);

	//検出点の描画
	image.copyTo(draw_image);
	if(detect)
	{
		//サブピクセル精度
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


//プロジェクタ画像上の交点座標を求める
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


//******外部にさらす用********//

DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
{
	return static_cast<void *>(new ProjectorEstimation(camWidth, camHeight, proWidth, proHeight, backgroundImgFile, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset));	
}

//パラメータファイル、3次元復元ファイル読み込み
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

//プロジェクタ位置推定コア呼び出し
DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, 
																double _initR[], double _initT[], double _dstR[], double _dstT[],
																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//カメラ画像をMatに復元
	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC3, cam_data);

	cv::Mat initR = (cv::Mat_<double>(3,3) << _initR[0], _initR[1], _initR[2], _initR[3], _initR[4], _initR[5], _initR[6], _initR[7], _initR[8] );
	cv::Mat initT = (cv::Mat_<double>(3,1) << _initT[0], _initT[1], _initT[2]);
	cv::Mat dstR, dstT;
	cv::Mat cam_drawimg = cam_img.clone();
	cv::Mat proj_drawing = pe->proj_img.clone();

	bool result = false;

	//位置推定メソッド呼び出し
	if(mode == 3)//チェッカパターン検出による推定
		result = pe->findProjectorPose(cam_img, initR, initT, dstR, dstT, cam_drawimg, proj_drawing);
	else         //コーナー検出による推定
		result = pe->findProjectorPose_Corner(cam_img, pe->proj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, mode, cam_drawimg, proj_drawing);

	if(result)
	{
		//推定結果を格納
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

	//コーナー検出結果表示
	cv::Mat resize_cam, resize_proj;
	cv::resize(cam_drawimg, resize_cam, cv::Size(), 0.5, 0.5);
	cv::imshow("Camera detected corners", resize_cam);
	cv::resize(proj_drawing, resize_proj, cv::Size(), 0.5, 0.5);
	cv::imshow("Projector detected corners", resize_proj);

	return result;
}

