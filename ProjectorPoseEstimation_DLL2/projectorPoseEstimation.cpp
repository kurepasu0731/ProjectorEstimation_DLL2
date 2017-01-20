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
bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &error,
												   int dotsCount, int dots_data[],
												   //int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, 
												   double thresh, 
												   int mode,
												   bool isKalman, bool isPredict,
												   //double C, int dotsMin, int dotsMax, float resizeScale,
												   cv::Mat &draw_camimage, cv::Mat &draw_projimage)
{
	//処理時間計測
	CFileTime cTimeStart_, cTimeEnd_;
	CFileTimeSpan cTimeSpan_;

	//cTimeStart_ = CFileTime::GetCurrentTime();// 現在時刻

	//startTic();

	bool detect_cam = false;
	if(mode == 4)
	{
		//cv::Mat src;
		//cv::cvtColor(camframe, src, CV_BGR2GRAY);
		////src = camframe.clone(); //PGR
		//detect_cam = getDots(src, camcorners, C, dotsMin, dotsMax, resizeScale, draw_camimage);

		//ドット配列をvectorにする
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
		//カメラ画像上のコーナー検出
//		detect_cam = getCorners(camframe, camcorners, camMinDist, camCornerNum, draw_camimage);
		//プロジェクタ画像上のコーナー検出
		//bool detect_proj = getCorners(projframe, projcorners, projMinDist, projCornerNum, draw_projimage); //projcornersがdraw_projimage上でずれるのは、歪み除去してないから
	}
	//stopTic("conerDetect");

	//コーナー検出できたら、位置推定開始
	if(detect_cam && detect_proj)
	{
		//startTic();

		// 対応点の歪み除去
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
		if(mode == 1 || mode == 4)//こっちしか機能してない
			//result = calcProjectorPose_Corner1(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner1(undistort_imagePoint, projcorners, thresh, isKalman, isPredict, initialR, initialT, _dstR, _dstT, _error, draw_camimage, draw_projimage);
		else if(mode == 2)
			//result = calcProjectorPose_Corner2(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);
			result = calcProjectorPose_Corner2(undistort_imagePoint, projcorners, initialR, initialT, _dstR, _dstT, draw_camimage, draw_projimage);

		_dstR.copyTo(dstR);
		_dstT.copyTo(dstT);
		_error.copyTo(error);

		//stopTic("calcConer1");

		//cTimeEnd_ = CFileTime::GetCurrentTime();           // 現在時刻
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

//計算部分
int ProjectorEstimation::calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh, bool isKalman, bool isPredict,
																		cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &error, cv::Mat &draw_camimage, cv::Mat &chessimage)
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
			if(0 <= image_x && image_x < camera->width && 0 <= image_y && image_y < camera->height &&
				reconstructPoints[index].x != -1.0f && reconstructPoints[index].y != -1.0f && reconstructPoints[index].z != -1.0f)
			{
				//マスク領域(ドラえもん部分)のコーナー点は除外
				if(CameraMask.data[index * 3 + 0] != 0 && CameraMask.data[index * 3 + 1] != 0 && CameraMask.data[index * 3 + 2] != 0 )
				{
					reconstructPoints_valid.emplace_back(reconstructPoints[index]);
					imagePoints_valid.emplace_back(imagePoints[i]);
				}
			}
			else
			{
				//カメラ画像に描画
				cv::circle(draw_camimage, cv::Point(image_x, image_y), 2, cv::Scalar(0, 255, 0), 3); //緑
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

		//stopTic("Kmeans");

		//対応順に3次元点を整列する
		std::vector<cv::Point3f> reconstructPoints_order;
		//対応付けられてるカメラ画像点も整列
		std::vector<cv::Point2f> imagePoints_order;
		//有効なプロジェクタ画像上の対応点
		std::vector<cv::Point2f> projPoints_valid;

		//startTic();

		///////↓↓対応点の選択↓↓///////

		//**指数平滑用定数**//
		double a = 0.45; //a > 1 -> 過去ほど重く　a < 1 ->過去ほど軽く

		//対応点間の距離が閾値以上離れていたら、外れ値として除去
		for(int i = 0; i < projPoints.size(); i++)
		{

			//**過去preframesizeフレームの加重平均**//
			double distAve = 0;
			//arrayが30貯まっていてかつ、それらの重みつけ平均が閾値以上なら、対応点として追加
			if(preDistsArrays[i].size() == preframesize)
			{
				//重みつけ平均
				for(int j = 0; j < preframesize; j++)
				{
					distAve += ((j + 1) * preDistsArrays[i][j]); //新しいものほど重みを重くする(<-こっちのほうがよさそう)
					//distAve += ((preframesize - j) * preDistsArrays[i][j]); //新しいものほど重みを軽くする
				}
				distAve /= sum;
			}

			////**指数平滑法**//
			//double expodist = 0.0;
			//double distance = sqrt(pow(projPoints[i].x - ppt[indices[i][0]].x, 2) + pow(projPoints[i].y - ppt[indices[i][0]].y, 2));
			////t=0のときはdt^ = dt
			//if(preExpoDists[i] == 0.0)
			//{
			//	expodist = distance;
			//}
			//else
			//{
			//	//expodist = a * distance + (1 - a) * preExpoDists[i];//普通の指数平滑法
			//	expodist = a * distance + (1/(1 - a)) * preExpoDists[i]; //過去ほど重くする逆指数平滑法
			//	expodist /= 100; //なんか違うような気がする
			//}
			//preExpoDists[i] = expodist;

			//std::string logSum = "expo:" + std::to_string(preExpoDists[i]);
			//debug_log(logSum);
			//debug_log(std::to_string(preDistsArrays[i].size()));
			//std::string logAve ="[" + std::to_string(i) + "]: " + std::to_string(distAve);
			//debug_log(logAve);

//			double distance = sqrt(pow(projPoints[i].x - ppt[indices[i][0]].x, 2) + pow(projPoints[i].y - ppt[indices[i][0]].y, 2));
//			if( distance <= thresh + 10 || preDists[i] <= thresh)//現フレームでの対応点間距離または前フレームでの距離が閾値以下ならば ->ここもっと時間軸で重み付けとかしたら良くなりそう
//			if( distance <= thresh) //単純
//			if(expodist <= thresh)//指数平滑法
			if(distAve <= thresh && preDistsArrays[i].size() == preframesize)//加重平均
			{
				reconstructPoints_order.emplace_back(reconstructPoints_valid[indices[i][0]]);
				imagePoints_order.emplace_back(imagePoints_valid[indices[i][0]]);
				projPoints_valid.emplace_back(projPoints[i]);
			}
			////preDistsの更新
			//preDists[i] = distance;
		}

		///////↑↑対応点の選択↑↑///////

		//stopTic("serect:");

		//はじかれた対応点も全部描画
		for(int i = 0; i < projPoints.size(); i++)
		{	
			cv::Point pp = cv::Point((int) (projPoints[i].x + 0.5f), (int) (projPoints[i].y + 0.5f));
			cv::Point cp = cv::Point((int) (ppt[indices[i][0]].x + 0.5f), (int) (ppt[indices[i][0]].y + 0.5f));

			//プロジェクタ画像のコーナー点を描画(対応ついてないのも全部)
			cv::circle(chessimage, pp, 1, cv::Scalar(0, 0, 255), 3); //小さめの赤
			cv::putText(chessimage,std::to_string(i), pp, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255));

			//カメラ画像のコーナー点をプロジェクタ画像へ射影したものの描画(対応ついてないのも全部)
			//描画(プロジェクタ画像)
			cv::circle(chessimage, cp, 1, cv::Scalar(255, 0, 0), 3); //小さめの青
			cv::putText(chessimage,std::to_string(i), cp, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0));

			//線で結ぶ
			cv::line(chessimage, pp, cp, cv::Scalar(0, 255, 0), 2);//緑
		}

		if(reconstructPoints_order.size() > 20) //10点以上残れば(最低6点?)
		{
			//startTic();

			//パラメータを求める(全点で)			
			cv::Mat _dstR, _dstT;
			//int result = calcParameters(projPoints_valid, reconstructPoints_order, initialR, initialT, _dstR, _dstT);
			//パラメータを求める(RANSAC)
			int result = calcParameters_RANSAC(projPoints_valid, reconstructPoints_order, initialR, initialT, 10, thresh, _dstR, _dstT);

			//stopTic("calclate");

			if(isKalman)
			{
				//startTic();

				//--補正あり--//
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

			///////↓↓動き予測↓↓///////
			// 回転
			glm::mat4 rotation( _dstR.at<double>(0, 0), _dstR.at<double>(1,0), _dstR.at<double>(2,0), 0.0f,
								_dstR.at<double>(0,1), _dstR.at<double>(1,1), _dstR.at<double>(2,1), 0.0f,
								_dstR.at<double>(0,2), _dstR.at<double>(1,2), _dstR.at<double>(2,2), 0.0f,
								0.f, 0.f, 0.f, 1.f);
			timer.stop();
			predict_point->addData(timer.MSec(), cv::Point3f(_dstT.at<double>(0, 0), _dstT.at<double>(1, 0), _dstT.at<double>(2, 0)));
			predict_quat->addData(timer.MSec(), glm::quat_cast(rotation));

			// 投影予測位置
			cv::Point3f predict_point2 = predict_point->calcYValue(timer.MSec()+trackingTime);

			// 遅延補償した位置姿勢
			glm::mat4 predictPose = glm::mat4_cast(predict_quat->calcYValue(timer.MSec()+trackingTime));
			predictPose[3][0] = predict_point2.x;
			predictPose[3][1] = predict_point2.y;
			predictPose[3][2] = predict_point2.z;

			// 予測フィルタ使用時
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

			///////↑↑動き予測↑↑///////

			////マスクをかける
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

			//対応点の様子を描画
			vector<cv::Point2d> projection_P;
			vector<double> errors;
			double aveError = 0; //平均再投影
			calcReprojectionErrors(projPoints_valid, reconstructPoints_order, _dstR, _dstT, projection_P, errors);
			//calcReprojectionErrors(projPoints_valid, reconstructPoints_order, _dstR, _dstT_kf, projection_P, errors);


			//有効な対応点の描画(R,t推定後)
			for(int i = 0; i < projPoints_valid.size(); i++)
			{
				cv::Point pp = cv::Point((int) (projPoints_valid[i].x + 0.5f), (int) (projPoints_valid[i].y + 0.5f));
				cv::Point cp = cv::Point((int) (projection_P[i].x + 0.5f), (int) (projection_P[i].y + 0.5f));
				cv::Point icp = cv::Point((int) (imagePoints_order[i].x + 0.5f), (int) (imagePoints_order[i].y + 0.5f));

				//描画(プロジェクタ画像)
				cv::circle(chessimage,pp, 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
				cv::circle(chessimage, cp, 5, cv::Scalar(255, 0, 0), 3);//カメラ(予測あり)は青
				//描画(カメラ画像)
				cv::circle(draw_camimage, icp, 1, cv::Scalar(255, 0, 0), 3); //対応つけられてるのは青に

				//線で結ぶ
				cv::line(chessimage, pp, cp, cv::Scalar(255, 0, 255), 4);//ピンク(太)

				aveError += errors[i];
			}

			aveError /= errors.size();
			////プロジェクタ画像の対応点が何％対応付けられているかの割合(％)
			//double percent = (projPoints_valid.size() * 100) / projPoints.size();

			//startTic();

			///////↓↓対応点間距離の更新↓↓///////
			//全点の対応点との距離(preDists)を更新
			//4*4行列にする
			cv::Mat _dstRt = (cv::Mat_<double>(4, 4) << _dstR.at<double>(0,0), _dstR.at<double>(0,1), _dstR.at<double>(0,2), _dstT.at<double>(0,0),
																			_dstR.at<double>(1,0), _dstR.at<double>(1,1), _dstR.at<double>(1,2), _dstT.at<double>(1,0),
																			_dstR.at<double>(2,0), _dstR.at<double>(2,1), _dstR.at<double>(2,2), _dstT.at<double>(2,0),
																			0, 0, 0, 1);
			for(int i = 0; i < projPoints.size(); i++)
			{
				// 2次元(プロジェクタ画像)平面へ投影
				cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[indices[i][0]].x, reconstructPoints_valid[indices[i][0]].y, reconstructPoints_valid[indices[i][0]].z, 1);
				cv::Mat dst_p = projK_34 * _dstRt * wp;
				cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				double distance = sqrt(pow(projPoints[i].x - pt.x, 2) + pow(projPoints[i].y - pt.y, 2));
				preDists[i] = distance;

				//arraysへの値の代入(MAXが30で、後ろからいれる・前のを押し出す)
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
			///////↑↑対応点間距離の更新↑↑///////

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

			//距離だけ初期値で算出
			for(int i = 0; i < projPoints.size(); i++)
			{
				// 2次元(プロジェクタ画像)平面へ投影
				cv::Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_valid[indices[i][0]].x, reconstructPoints_valid[indices[i][0]].y, reconstructPoints_valid[indices[i][0]].z, 1);
				cv::Mat dst_p = projK_34 * initialRt * wp;
				cv::Point2d pt(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				double distance = sqrt(pow(projPoints[i].x - pt.x, 2) + pow(projPoints[i].y - pt.y, 2));
				preDists[i] = distance;

				//arraysへの値の代入(MAXが30で、後ろからいれる・前のを押し出す)
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

	//高精度化
	cv::cornerSubPix(gray_img, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

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
	info = lm.minimize(initial);//->ここが重い
    
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

//Ceres Solver Version(マイナス30msくらいになった！！！)
int ProjectorEstimation::calcParameters_Ceres(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
{
	////回転行列からクォータニオンにする
	//cv::Mat initialR_tr = initialR.t();//関数の都合上転置
	//double w, x, y, z;
	//transformRotMatToQuaternion(x, y, z, w, initialR_tr.at<double>(0, 0), initialR_tr.at<double>(0, 1), initialR_tr.at<double>(0, 2), 
	//															initialR_tr.at<double>(1, 0), initialR_tr.at<double>(1, 1), initialR_tr.at<double>(1, 2), 
	//															initialR_tr.at<double>(2, 0), initialR_tr.at<double>(2, 1), initialR_tr.at<double>(2, 2)); 		
	//double camera[6] = {x, y, z, initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0) };
	
	//回転行列からオイラーにする
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

	//出力
	//回転
	cv::Mat dst_euler = (cv::Mat_<double>(3, 1) << camera[0], camera[1], camera[2]);
	cv::Mat _dstR = kf.euler2rot(dst_euler);

	//並進
	cv::Mat _dstT = (cv::Mat_<double>(3, 1) << camera[3], camera[4], camera[5]);

	_dstR.copyTo(dstR);
	_dstT.copyTo(dstT);

	return summary.IsSolutionUsable();
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

//対応点からRとTの算出(RANSAC)
int ProjectorEstimation::calcParameters_RANSAC(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT,int num, float thresh, cv::Mat& dstR, cv::Mat& dstT)
{
		//inlierの割合
		double maxpercentage = 0.0;
		//最大スコア
		int maxscore = 0;
		//inlier対応点
		std::vector<cv::Point3f> inlier_P;
		std::vector<cv::Point2f> inlier_p;

		//繰り返し回数
		int iterate = 0;

		//ランダムに選んでくる点
		std::vector<cv::Point3f> random_P;
		std::vector<cv::Point2f> random_p;

		//再投影誤差
		vector<cv::Point2d> projection_P;
		vector<double> errors;

		while(iterate < 10)
		{
			//クリア
			random_P.clear();
			random_p.clear();

			//1. ランダムにnum点選ぶ
			get_random_points(num, src_p, src_P, random_p, random_P);

			//2. num点でパラメータを求める			
			cv::Mat preR, preT;//仮パラメータ
			int result = calcParameters_Ceres(random_p, random_P, initialR, initialT, preR, preT);
			//int result = calcParameters(random_p, random_P, initialR, initialT, preR, preT);

			//debug_log(std::to_string(result));

			if(result > 0)
			{
				projection_P.clear();
				errors.clear();

				//3. 全点で再投影誤差を求める
				calcReprojectionErrors(src_p, src_P, preR, preT, projection_P, errors);

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

		//5. inlierで再度パラメータを求める
		cv::Mat final_R, final_T;
		//debug_log("final");
		int result = calcParameters_Ceres(inlier_p, inlier_P, initialR, initialT, final_R, final_T); //->ここめっちゃかかる(21~26ms)
		//int result = calcParameters(inlier_p, inlier_P, initialR, initialT, final_R, final_T); //->ここめっちゃかかる(21~26ms)

		//対応点の様子を描画
		projection_P.clear();
		errors.clear();
		double aveError = 0; //平均再投影

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

//処理時間計測用・DebugLog表示用
void ProjectorEstimation::stopTic(std::string label)
{
		cTimeEnd = CFileTime::GetCurrentTime();           // 現在時刻
		cTimeSpan = cTimeEnd - cTimeStart;
		//debug_log(log);
		//debug_log(std::to_string(cTimeSpan.GetTimeSpan()/10000));
		std::string timelog = label +": " + std::to_string(cTimeSpan.GetTimeSpan()/10000);
		debug_log(timelog);
}

//**ランダムドットマーカー用**//	
//csvファイルから円の座標を読み込む
bool ProjectorEstimation::loadDots(std::vector<cv::Point2f> &corners, cv::Mat &drawimage)
{
	string filename = "Calibration/dots.csv";

    //ファイルの読み込み
    ifstream ifs(filename);
    if(!ifs){
        return false;
    }

    //csvファイルを1行ずつ読み込む
    string str;
    while(getline(ifs,str)){
        string token;
        istringstream stream(str);

		//x座標
		getline(stream,token,',');
		int x = std::stoi(token);
		//y座標
		getline(stream,token,',');
		int y = std::stoi(token);

		corners.emplace_back(cv::Point2f(x, y));

	}
	return true;
}
//ドット検出
bool ProjectorEstimation::getDots(cv::Mat &src, std::vector<cv::Point2f> &dots, double C, int dots_thresh_min, int dots_thresh_max, float resizeScale, cv::Mat &drawimage)
{
	dots.clear();
	//リサイズ
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(), resizeScale, resizeScale);
	//適応的閾値処理
	cv::adaptiveThreshold(resized, resized, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, C);
	//膨張処理
	cv::dilate(resized, resized, cv::Mat());
	//原寸大に戻す
	cv::Mat ptsImg = cv::Mat::zeros( camera->height, camera->width, CV_8UC1); //原寸大表示用
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
	// OpenGL用に予めRGB用のデータ作成
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
	// OpenGL用に予めRGB用のデータ作成
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

