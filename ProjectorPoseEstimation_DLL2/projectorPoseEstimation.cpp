#include "projectorPoseEstimation.h"
#include "DebugLogWrapper.h"


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

	//後で使うプロジェクタの内部行列
	projK_34 = (cv::Mat_<double>(3, 4) << projector->cam_K.at<double>(0,0),projector->cam_K.at<double>(0,1), projector->cam_K.at<double>(0,2), 0,
						        projector->cam_K.at<double>(1,0), projector->cam_K.at<double>(1,1), projector->cam_K.at<double>(1,2), 0,
								projector->cam_K.at<double>(2,0), projector->cam_K.at<double>(2,1), projector->cam_K.at<double>(2,2), 0);

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
bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat camframe, const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, 
												   int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, double thresh, int mode,
												   cv::Mat &draw_camimage, cv::Mat &draw_projimage)
{
	//draw用(カメラ)
	draw_camimage = camframe.clone();

	//カメラ画像上のコーナー検出
	bool detect_cam = getCorners(camframe, camcorners, camMinDist, camCornerNum, draw_camimage);
	//プロジェクタ画像上のコーナー検出
	//bool detect_proj = getCorners(projframe, projcorners, projMinDist, projCornerNum, draw_projimage); //projcornersがdraw_projimage上でずれるのは、歪み除去してないから

	//コーナー検出できたら、位置推定開始
	if(detect_cam && detect_proj)
	{
		// 対応点の歪み除去
		std::vector<cv::Point2f> undistort_imagePoint;
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

		int result = 0;
		if(mode == 1)//こっちしか機能してない
			//result = calcProjectorPose_Corner1(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner1(undistort_imagePoint, projcorners, thresh, initialR, initialT, _dstR, _dstT, draw_camimage, draw_projimage);
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

//計算部分
int ProjectorEstimation::calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh,
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage)
{

		//3次元座標が取れた対応点のみを抽出してからLM法に入れる
		std::vector<cv::Point3f> reconstructPoints_valid;
		//対応付けられてるカメラ画像点
		std::vector<cv::Point2f> imagePoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * camera->width + image_x;
			//-1はプロジェクタ投影領域外エリアを意味する
			if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height && reconstructPoints[index].x != -1)
			{
				//マスク領域(ドラえもん部分)のコーナー点は除外
				if(CameraMask.data[index * 3 + 0] != 0 && CameraMask.data[index * 3 + 1] != 0 && CameraMask.data[index * 3 + 2] != 0 )
				{
					reconstructPoints_valid.emplace_back(reconstructPoints[index]);
					imagePoints_valid.emplace_back(imagePoints[i]);
				}
			}
		}

		///////↓↓最近傍探索で対応を求める↓↓///////

		// 2次元(プロジェクタ画像)平面へ投影
		std::vector<cv::Point2d> ppt;
		//4*4行列にする
		cv::Mat Rt = (cv::Mat_<double>(4, 4) << initialR.at<double>(0,0), initialR.at<double>(0,1), initialR.at<double>(0,2), initialT.at<double>(0,0),
																		initialR.at<double>(1,0), initialR.at<double>(1,1), initialR.at<double>(1,2), initialT.at<double>(1,0),
																		initialR.at<double>(2,0), initialR.at<double>(2,1), initialR.at<double>(2,2), initialT.at<double>(2,0),
																		0, 0, 0, 1);
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2次元(プロジェクタ画像)平面へ投影
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			ppt.emplace_back(pt);
		}
		//最近傍探索 X:カメラ点　Y:プロジェクタ点
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
		//indices[Yのインデックス][0] = 対応するXのインデックス
		index.knnSearch(mat_Y,
								indices,
								dists,
								1, // k of knn
								flann::SearchParams() );
		///////↑↑最近傍探索で対応を求める↑↑///////

#if 0
		//---RANSAC---//
		//対応する3次元点
		std::vector<cv::Point3f> Points3D;
		for(int i = 0; i < projPoints.size(); i++)
			Points3D.emplace_back(reconstructPoints_valid[indices[i][0]]);

		//inlierの割合
		double maxpercentage = 0.0;
		//最大スコア
		int maxscore = 0;
		//inlier対応点
		std::vector<cv::Point3f> inlier_P;
		std::vector<cv::Point2f> inlier_p;

		//繰り返し回数
		int iterate = 0;

		//for(int i = 0; i < 10; i++)
		while(iterate <= 10)
		{
			////クリア
			//inlier_P.clear();
			//inlier_p.clear();

			//1. ランダムに6点選ぶ
			std::vector<cv::Point3f> random_P;
			std::vector<cv::Point2f> random_p;
			get_random_points(20, projPoints, Points3D, random_p, random_P);

			//2. 6点でパラメータを求める			
			cv::Mat preR, preT;//仮パラメータ
			int result = calcParameters(random_p, random_P, initialR, initialT, preR, preT);

			debug_log(std::to_string(result));

			if(result > 0)
			{
				//3. 全点で再投影誤差を求める
				vector<cv::Point2d> projection_P;
				vector<double> errors;
				calcReprojectionErrors(projPoints, Points3D, preR, preT, projection_P, errors);

				//4. 再投影誤差が閾値以下だったものの割合が99％以下だったら、続行
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
					//クリア
					inlier_P.clear();
					inlier_p.clear();

					for(int i = 0; i < errors.size(); i++)
					{
						if(errors[i] <= thresh)
						{
							inlier_p.emplace_back(projPoints[i]);
							inlier_P.emplace_back(Points3D[i]);
						}
					}
					maxpercentage = score * 100 / projPoints.size();
					maxscore = score;
				}

				if(maxpercentage >= 90) break;

				iterate++;

			}
		}

		//5. inlierで再度パラメータを求める
		cv::Mat final_R, final_T;
		int result = calcParameters(inlier_p, inlier_P, initialR, initialT, final_R, final_T);

		//対応点の様子を描画
		vector<cv::Point2d> projection_P;
		vector<double> errors;
		double aveError = 0; //平均再投影
		calcReprojectionErrors(inlier_p, inlier_P, final_R, final_T, projection_P, errors);
		for(int i = 0; i < inlier_p.size(); i++)
		{
			cv::circle(chessimage, inlier_p[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
			cv::circle(chessimage, projection_P[i], 5, cv::Scalar(255, 0, 0), 3);//カメラ(予測なし)は青
			aveError += errors[i];
		}
		aveError /= errors.size();

		std::string logAve = "aveError: ";
		std::string logAve2 =std::to_string(aveError);
		debug_log(logAve);
		debug_log(logAve2);
		std::string logPer = "valid points: ";
		std::string logPer2 = std::to_string(maxpercentage);
		debug_log(logPer);
		debug_log(logPer2);

		final_R.copyTo(dstR);
		final_T.copyTo(dstT);

		return result;

		//--RANSACおわり---//
#endif


#if 1//きれいに書き直したver(5msくらい遅い)
		//ロバスト推定
		//対応点の重み
		//std::vector<double> weight;

		//対応順に3次元点を整列する
		std::vector<cv::Point3f> reconstructPoints_order;
		//対応付けられてるカメラ画像点も整列
		std::vector<cv::Point2f> imagePoints_order;
		//有効なプロジェクタ画像上の対応点
		std::vector<cv::Point2f> projPoints_valid;
		
		//対応点間の距離が閾値以上離れていたら、外れ値として除去
		for(int i = 0; i < projPoints.size(); i++){
			double distance = dists[i][0];
			//double distance = sqrt(pow(projPoints[i].x - ppt[indices[i][0]].x, 2) + pow(projPoints[i].y - ppt[indices[i][0]].y, 2));
			if( distance <= thresh)
			{
				reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
				imagePoints_order.emplace_back(imagePoints_valid[indices[i][0]]);
				projPoints_valid.emplace_back(projPoints[i]);

				//double w = pow((1 - pow((distance / thresh), 2)), 2);
				//weight.emplace_back(w);
			}
			//else
			//{
			//	reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
			//	imagePoints_order.emplace_back(imagePoints_valid[indices[i][0]]);
			//	//projPoints_valid.emplace_back(projPoints[i]);
			//	//thresh外は重み0
			//	weight.emplace_back(0.0);
			//}
		}



		if(reconstructPoints_order.size() > 0)
		{

			//パラメータを求める			
			cv::Mat _dstR, _dstT;
			int result = calcParameters(projPoints_valid, reconstructPoints_order, initialR, initialT, _dstR, _dstT);

			//対応点の様子を描画
			vector<cv::Point2d> projection_P;
			vector<double> errors;
			double aveError = 0; //平均再投影
			calcReprojectionErrors(projPoints_valid, reconstructPoints_order, _dstR, _dstT, projection_P, errors);

			for(int i = 0; i < projPoints_valid.size(); i++)
			{
				cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
				cv::circle(chessimage, projection_P[i], 5, cv::Scalar(255, 0, 0), 3);//カメラ(予測なし)は青
				//描画(カメラ画像)
				cv::circle(draw_camimage, imagePoints_order[i], 1, cv::Scalar(255, 0, 0), 3); //対応つけられてるのは青に
				aveError += errors[i];
			}
			aveError /= errors.size();
			//プロジェクタ画像の対応点が何％対応付けられているかの割合(％)
			double percent = (projPoints_valid.size() * 100) / projPoints.size();

			_dstR.copyTo(dstR);
			_dstT.copyTo(dstT);

			//std::string logAve = "aveError: ";
			//std::string logAve2 =std::to_string(aveError);
			//std::string logPer = "valid points: ";
			//std::string logPer2 = std::to_string(percent);
			//debug_log(logAve);
			//debug_log(logAve2);
			//debug_log(logPer);
			//debug_log(logPer2);

			return result;

#if 0//べた書きver(5ms速い)
			//回転行列からクォータニオンにする
			cv::Mat initialR_tr = initialR.t();//関数の都合上転置
			double w, x, y, z;
			transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), 
																	  initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), 
																	  initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
			int n = 6; //変数の数
			int info;

			VectorXd initial(n);
			initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);

			misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_order, projK_34);
    
			NumericalDiff<misra1a_functor> numDiff(functor);
			LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

			//最適化
			info = lm.minimize(initial);
    
			//出力
			//回転
			Quaterniond q(0, initial[0], initial[1], initial[2]);
			q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
			q.normalize ();
			MatrixXd qMat = q.toRotationMatrix();
			cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
			//並進
			//--予測なし--//
			cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
			////--予測あり--//
			//cv::Mat translation_measured = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
			//cv::Mat measurement(3, 1, CV_64F);
			//kf.fillMeasurements(measurement, translation_measured);
			//// Instantiate estimated translation and rotation
			//cv::Mat translation_estimated(3, 1, CV_64F);
			//// update the Kalman filter with good measurements
			//kf.updateKalmanfilter(measurement, translation_estimated);
			//cv::Mat _dstT_kf = (cv::Mat_<double>(3, 1) << translation_estimated.at<double>(0, 0), translation_estimated.at<double>(1, 0), translation_estimated.at<double>(2, 0));


			//対応点の投影誤差
			double aveError = 0;
			//対応点の様子を描画
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																			_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																			_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																			0, 0, 0, 1);
			//cv::Mat Rt_kf = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT_kf.at<double>(0,0),
			//																_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT_kf.at<double>(1,0),
			//																_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT_kf.at<double>(2,0),
			//																0, 0, 0, 1);
			for(int i = 0; i < reconstructPoints_order.size(); i++)
			{
				// 2次元(プロジェクタ画像)平面へ投影
				cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_order[i].x, reconstructPoints_order[i].y, reconstructPoints_order[i].z, 1);
				//4*4行列にする
				//--予測なし--//
				cv::Mat dst_p = projK_34 * Rt * wp;
				cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				////--予測あり--//
				//cv::Mat dst_p_kf = projK_34 * Rt_kf * wp;
				//cv::Point2d pt_kf(dst_p_kf.at<double>(0,0) / dst_p_kf.at<double>(2,0), dst_p_kf.at<double>(1,0) / dst_p_kf.at<double>(2,0));
				//描画(プロジェクタ画像)
				cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
				cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//カメラ(予測なし)は青
				//cv::circle(chessimage, pt_kf, 5, cv::Scalar(0, 255, 0), 3);//カメラ(予測あり)は緑
				//描画(カメラ画像)
				cv::circle(draw_camimage, imagePoints_order[i], 1, cv::Scalar(255, 0, 0), 3); //対応つけられてるのは青に
				//対応点の再投影誤差算出
				double error = sqrt(pow(pt.x - projPoints_valid[i].x, 2) + pow(pt.y - projPoints_valid[i].y, 2));
				aveError += error;
			}

			//対応点の平均再投影誤差
			aveError /= projPoints_valid.size();
			//std::cout << "reprojection error ave : " << aveError << std::endl;
			//プロジェクタ画像の対応点が何％対応付けられているかの割合(％)
			double percent = (projPoints_valid.size() * 100) / projPoints.size();

			//std::string logAve = "aveError: ";
			//std::string logAve2 =std::to_string(aveError);
			//std::string logPer = "valid points: ";
			//std::string logPer2 = std::to_string(percent);
			//debug_log(logAve);
			//debug_log(logAve2);
			//debug_log(logPer);
			//debug_log(logPer2);

			////閾値の更新
			//if(thresh > 10)
			//{
			//	//単位はmm
			//	if(aveError <= 20 && percent >= 80)//止まった判定
			//	{
			//		thresh = 10;
			//	}
			//}
			//else
			//{
			//	if(aveError >= 5 && percent <= 30)//動きだし判定
			//	{
			//		thresh = 50;
			//	}
			//}

			_dstR.copyTo(dstR);
			_dstT.copyTo(dstT);
			//_dstT_kf.copyTo(dstT);

			std::cout << "info: " << info << std::endl;
			return info;
#endif
		}
		else return 0;
#endif
}



//計算部分(最近傍探索を3次元点の方に合わせる)
int ProjectorEstimation::calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints,
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage)
{
		////回転行列から回転ベクトルにする
		//cv::Mat initRVec(3, 1,  CV_64F, cv::Scalar::all(0));
		//Rodrigues(initialR, initRVec);
		//回転行列からクォータニオンにする
		cv::Mat initialR_tr = initialR.t();//関数の都合上転置
		double w, x, y, z;
		transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		

		int n = 6; //変数の数
		int info;

		VectorXd initial(n);
		initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);

		//3次元座標が取れた対応点のみを抽出してからLM法に入れる
		std::vector<cv::Point3f> reconstructPoints_valid;
		//対応付けられてるカメラ画像点
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

		// 2次元(プロジェクタ画像)平面へ投影
		//std::vector<cv::Point2f> ppt;
		//cv::projectPoints(reconstructPoints_valid, initialR, initTVec, projector.cam_K, cv::Mat(), ppt); 
		std::vector<cv::Point2d> ppt;
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2次元(プロジェクタ画像)平面へ投影
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			//4*4行列にする
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << initialR.at<double>(0,0), initialR.at<double>(0,1), initialR.at<double>(0,2), initialT.at<double>(0,0),
																		  initialR.at<double>(1,0), initialR.at<double>(1,1), initialR.at<double>(1,2), initialT.at<double>(1,0),
																		  initialR.at<double>(2,0), initialR.at<double>(2,1), initialR.at<double>(2,2), initialT.at<double>(2,0),
																		  0, 0, 0, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			ppt.emplace_back(pt);
		}

		///////↓↓最近傍探索で対応を求める↓↓///////

		//最近傍探索 X:カメラ点　Y:プロジェクタ点
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
		//indices[Xのインデックス][0] = 対応するYのインデックス
		index.knnSearch(mat_X,
								indices,
								dists,
								1, // k of knn
								flann::SearchParams() );

		//対応点の重み
		//std::vector<double> weight;
		//対応順に3次元点を整列する
		std::vector<cv::Point2f> projPoints_order;
		for(int i = 0; i < reconstructPoints_valid.size(); i++){
			projPoints_order.emplace_back(projPoints[indices[i][0]]);
			////今仮
			//weight.emplace_back(1.0);
		}

		misra1a_functor functor(n, projPoints_order.size(), projPoints_order, reconstructPoints_valid, projK_34);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

		///////↑↑最近傍探索で対応を求める↑↑///////

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
		//cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		//Rodrigues(dstRVec, dstR); //->src.copyTo(data)使って代入しないとダメ　じゃなくて　回転ベクトルを毎回正規化しないとダメ
		//回転
		Quaterniond q(0, initial[0], initial[1], initial[2]);
		q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
		q.normalize ();
		MatrixXd qMat = q.toRotationMatrix();
		cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
		//並進
		cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
		//cv::Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

		//対応点の投影誤差算出
		double aveError = 0;

		//対応点の様子を描画
		//std::vector<cv::Point2f> pt;
		//cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
		for(int i = 0; i < reconstructPoints_valid.size(); i++)
		{
			// 2次元(プロジェクタ画像)平面へ投影
			cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
			//4*4行列にする
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																		  _dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																		  _dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																		  0, 0, 0, 1);
			cv::Mat dst_p = projK_34 * Rt * wp;
			cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
			//描画
			cv::circle(chessimage, projPoints_order[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
			cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//カメラは青
			//描画(カメラ画像)
			cv::circle(draw_camimage, imagePoints_valid[i], 1, cv::Scalar(255, 0, 0), 3); //対応つけられてるのは青に

			double error = sqrt(pow(pt.x - projPoints_order[i].x, 2) + pow(pt.y - projPoints_order[i].y, 2));
			aveError += error;
		}

		////重心も描画
		//cv::Point2f imageWorldPointAve;
		//cv::Point2f projAve;
		//calcAveragePoint(reconstructPoints_valid, projPoints, dstRVec, dstTVec,imageWorldPointAve, projAve);
		//cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
		//cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

		std::cout << "reprojection error ave : " << (double)(aveError / projPoints.size()) << std::endl;

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);

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
	cv::goodFeaturesToTrack(gray_img, corners, num, 0.01, minDistance);

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
bool ProjectorEstimation::findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage){

	//チェッカパターン検出(カメラ画像は歪んだまま)
	bool detect = getCheckerCorners(cameraImageCorners, frame, draw_image);

	//検出できたら、位置推定開始
	if(detect)
	{
		// 対応点の歪み除去
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

//計算部分(Rの自由度3)
int ProjectorEstimation::calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage)
{
	//回転行列からクォータニオンにする
	cv::Mat initialR_tr = initialR.t();//関数の都合上転置
	double w, x, y, z;
	transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
	cv::Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

	int n = 6; //変数の数
	int info;

	VectorXd initial(n);
	initial << x, y, z, initTVec.at<double>(0, 0), initTVec.at<double>(1, 0), initTVec.at<double>(2, 0);

	//対応点の重み(ダミー)
	//std::vector<double> weight;

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
			//weight.emplace_back(1.0);
		}
	}

	misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projK_34);
    
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
	//回転
	Quaterniond q(0, initial[0], initial[1], initial[2]);
	q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
	q.normalize ();
	MatrixXd qMat = q.toRotationMatrix();
	cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
	//並進
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);

	//対応点の様子を描画
	//std::vector<cv::Point2f> pt;
	//cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
	//4*4行列にする
	cv::Mat Rt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																	_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																	_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																	0, 0, 0, 1);
	for(int i = 0; i < projPoints_valid.size(); i++)
	{
		// 2次元(プロジェクタ画像)平面へ投影
		cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[i].x, reconstructPoints_valid[i].y, reconstructPoints_valid[i].z, 1);
		cv::Mat dst_p = projK_34 * Rt * wp;
		cv::Point2f pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
		//描画
		cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
		cv::circle(chessimage, pt, 5, cv::Scalar(255, 0, 0), 3);//カメラは青
	}
	////重心も描画
	//cv::Point2f imageWorldPointAve;
	//cv::Point2f projAve;
	//calcAveragePoint(reconstructPoints_valid, projPoints_valid, dstRVec, dstTVec,imageWorldPointAve, projAve);
	//cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
	//cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青


	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);
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

///////////////////////////////////////////////
// 回転行列→クォータニオン変換
//
// qx, qy, qz, qw : クォータニオン成分（出力）
// m11-m33 : 回転行列成分
//
// ※注意：
// 行列成分はDirectX形式（行方向が軸の向き）です
// OpenGL形式（列方向が軸の向き）の場合は
// 転置した値を入れて下さい。

bool ProjectorEstimation::transformRotMatToQuaternion(
    double &qx, double &qy, double &qz, double &qw,
    double m11, double m12, double m13,
    double m21, double m22, double m23,
    double m31, double m32, double m33
) {
    // 最大成分を検索
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
        return false; // 引数の行列に間違いあり！

    // 最大要素の値を算出
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


//ランダムにnum点を抽出
void ProjectorEstimation::get_random_points(int num, vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, vector<cv::Point2f>& calib_p, vector<cv::Point3f>& calib_P){
	int i=0;
	//初期化
	calib_p.clear();
	calib_P.clear();

	//srand(time(NULL));    /* 乱数の初期化 */ 
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

//対応点からRとTの算出
int ProjectorEstimation::calcParameters(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT){
	//回転行列からクォータニオンにする
	cv::Mat initialR_tr = initialR.t();//関数の都合上転置
	double w, x, y, z;
	transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), 
																initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), 
																initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
		
	int n = 6; //変数の数
	int info;

	VectorXd initial(n);
	initial << x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0);

	misra1a_functor functor(n, src_p.size(), src_p, src_P, projK_34);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);

	//最適化
	info = lm.minimize(initial);
    
	//出力
	//回転
	Quaterniond q(0, initial[0], initial[1], initial[2]);
	q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
	q.normalize ();
	MatrixXd qMat = q.toRotationMatrix();
	cv::Mat _dstR = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
	//並進
	//--予測なし--//
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);

	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);

	return info;
}

//3次元点のプロジェクタ画像への射影と再投影誤差の計算
void ProjectorEstimation::calcReprojectionErrors(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat R, cv::Mat T, vector<cv::Point2d>& projection_P, vector<double>& errors){
	//対応点の様子を描画
	cv::Mat Rt = (cv::Mat_<double>(4, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), T.at<double>(0,0),
																	R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), T.at<double>(1,0),
																	R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), T.at<double>(2,0),
																	0, 0, 0, 1);
	for(int i = 0; i < src_P.size(); i++)
	{
		// 2次元(プロジェクタ画像)平面へ投影
		cv::Mat wp = (cv::Mat_<double>(4, 1) << src_P[i].x, src_P[i].y, src_P[i].z, 1);
		//4*4行列にする
		//--予測なし--//
		cv::Mat dst_p = projK_34 * Rt * wp;
		cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
		//対応点の再投影誤差算出
		double reprojectError = sqrt(pow(pt.x - src_p[i].x, 2) + pow(pt.y - src_p[i].y, 2));

		projection_P.emplace_back(pt);
		errors.emplace_back(reprojectError);
	}
}

//******外部にさらす用********//
/*
DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset, double _thresh)
{
	return static_cast<void *>(new ProjectorEstimation(camWidth, camHeight, proWidth, proHeight, backgroundImgFile, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset, _thresh));	
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
DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
																double _initR[], double _initT[], double _dstR[], double _dstT[],
																int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//カメラ画像をMatに復元
	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC3, cam_data);

	cv::Mat initR = (cv::Mat_<double>(3,3) << _initR[0], _initR[1], _initR[2], _initR[3], _initR[4], _initR[5], _initR[6], _initR[7], _initR[8] );
	cv::Mat initT = (cv::Mat_<double>(3,1) << _initT[0], _initT[1], _initT[2]);

	//1フレーム後の推定値
	cv::Mat dstR = cv::Mat::eye(3,3,CV_64F);
	cv::Mat dstT = cv::Mat::zeros(3,1,CV_64F);

	cv::Mat cam_drawimg = cam_img.clone();
	cv::Mat proj_drawing;

	bool result = false;

	//位置推定メソッド呼び出し
	if(mode == 3)//チェッカパターン検出による推定
	{	proj_drawing = pe->proj_img.clone();
		result = pe->findProjectorPose(cam_img, initR, initT, dstR, dstT, cam_drawimg, proj_drawing);
	}
	//コーナー検出による推定(プロジェクタ画像更新ver 4->1, 5->2)
	else if(mode == 4 || mode == 5)
	{
		//プロジェクタ画像をMatに復元
		cv::Mat prj_img(pe->projector->height, pe->projector->width, CV_8UC4, prj_data);
		//プロジェクタ画像はUnity側で生成されたので、反転とかする
		//BGR <-- ARGB 変換
		cv::Mat bgr_img, flip_prj_img;
		std::vector<cv::Mat> bgra;
		cv::split(prj_img, bgra);
		std::swap(bgra[0], bgra[3]);
		std::swap(bgra[1], bgra[2]);
		cv::cvtColor(prj_img, bgr_img, CV_BGRA2BGR);
		//x軸反転
		cv::flip(bgr_img, flip_prj_img, 0);

		proj_drawing = flip_prj_img.clone();

		result = pe->findProjectorPose_Corner(cam_img, flip_prj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, mode-3, cam_drawimg, proj_drawing);
	}
	//コーナー検出による推定(プロジェクタ画像更新しないver)
	else
	{	proj_drawing = pe->proj_img.clone();
		result = pe->findProjectorPose_Corner(cam_img, pe->proj_img, initR, initT, dstR, dstT, camCornerNum, camMinDist, projCornerNum, projMinDist, mode, cam_drawimg, proj_drawing);
	}

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
	//マスクをかける
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

//カメラ画像用マスクの作成
DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data)
{
	auto pe = static_cast<ProjectorEstimation*>(projectorestimation);

	//カメラ画像をMatに復元
	cv::Mat cam_img(pe->camera->height, pe->camera->width, CV_8UC4, cam_data);
	//プロジェクタ画像はUnity側で生成されたので、反転とかする
	//BGR <-- ARGB 変換
	cv::Mat bgr_img, flip_cam_img;
	std::vector<cv::Mat> bgra;
	cv::split(cam_img, bgra);
	std::swap(bgra[0], bgra[3]);
	std::swap(bgra[1], bgra[2]);
	cv::cvtColor(cam_img, bgr_img, CV_BGRA2BGR);
	//x軸反転
	cv::flip(bgr_img, flip_cam_img, 0);

	pe->CameraMask = flip_cam_img.clone();

	//一応保存
	cv::imwrite("CameraMask.png", pe->CameraMask);
}
*/


