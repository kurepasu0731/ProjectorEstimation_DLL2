#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"
#include "KalmanFilter.h"
#include "ErrorFunction.h"


#include <opencv2/opencv.hpp>
#include <random>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>

#include <atltime.h>

#include<fstream>
#include<iostream>
#include<string>
#include<sstream> //文字ストリーム

#include "LSM/LSMPoint3f.h"
#include "LSM/LSMQuatd.h"
#include "myTimer.h"


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;
	cv::Size checkerPattern;

	//カメラ画像上のコーナー点
	std::vector<cv::Point2f> camcorners;
	//プロジェクタ画像上のコーナー点
	std::vector<cv::Point2f> projcorners;

	//歪み除去後のカメラ画像のコーナー点
	std::vector<cv::Point2f> undistort_imagePoint;


	//3次元点(カメラ中心)LookUpテーブル
	//** index = カメラ画素(左上始まり)
	//** int image_x = i % CAMERA_WIDTH;
	//** int image_y = (int)(i / CAMERA_WIDTH);
	//** Point3f = カメラ画素の3次元座標(計測されていない場合は(-1, -1, -1))
	std::vector<cv::Point3f> reconstructPoints;

	//プロジェクタ画像
	cv::Mat proj_img, proj_undist;

	//3 * 4形式ののプロジェクタ内部行列
	cv::Mat projK_34;

	//カメラ画像(歪みなし)のマスク
	cv::Mat CameraMask;

	//カルマンフィルタ
	Kalmanfilter kf;

	//1フレーム前の対応点間距離
	std::vector<double> preDists;

	//指数平滑法用1フレーム前のdt^
	std::vector<double> preExpoDists;

	//過去preframesizeフレーム分の対応点間距離
	std::vector<std::vector<double>> preDistsArrays;
	//どのくらい過去の情報までみるかのフレーム数
	int preframesize;
	int sum;

	//動き予測関連//
	double trackingTime;		// トラッキングにかかる時間
	std::unique_ptr<LSMPoint3f> predict_point;	// 位置の最小二乗法
	std::unique_ptr<LSMQuatd> predict_quat;		// 姿勢の最小二乗法
	bool firstTime;								// 1回目かどうか
	MyTimer timer;				// 起動時間からの時間計測

	//処理時間計測
	CFileTime cTimeStart, cTimeEnd;
	CFileTimeSpan cTimeSpan;

	//--フラグ関係--//
	bool detect_proj; //プロジェクタ画像のコーナー点を検出したかどうか

	//コンストラクタ
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight, double trackingtime, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);
		checkerPattern = cv::Size(_checkerCol, _checkerRow);

		kf.initKalmanfilter(18, 6, 0, 0.03);//等速度

		//プロジェクタ画像読み込み,描画用画像作成
		proj_img = cv::imread(backgroundImgFile);
		proj_undist =  proj_img.clone();
		cv::undistort(proj_img, proj_undist, projector->cam_K, projector->cam_dist);


		////1フレーム前の対応点間距離の初期化
		//for(int i = 0; i < _checkerCol*_checkerRow; i++)
		//{
		//	preDists.emplace_back(0.0);
		//	preExpoDists.emplace_back(0.0);
		//}

		//std::vector<double> array;
		//array.clear();
		//for(int i = 0; i < _checkerCol*_checkerRow; i++)
		//{
		//	preDistsArrays.emplace_back(array);
		//}

		preframesize = 20; 
		//sum計算
		sum = 0;
		for(int i = 1; i <= preframesize; i++)
		{
			sum += i;
		}

		//コーナー検出の場合
		//TODO:プロジェクタ画像上のコーナー点を求めておく
		detect_proj = false;

		//動き予測関連//
		predict_point = std::unique_ptr<LSMPoint3f> (new LSMPoint3f(LSM));
		predict_quat = std::unique_ptr<LSMQuatd> (new LSMQuatd(LSM));
		predict_point->setForgetFactor(0.6);	// 忘却係数
		predict_quat->setForgetFactor(0.6);	// 忘却係数
		firstTime = true;
		trackingTime = trackingtime;//処理時間＋システム遅延
		timer = MyTimer();
		timer.start();
	};

	~ProjectorEstimation(){};

	//キャリブレーションファイル読み込み
	void loadProCamCalibFile(const std::string& filename);

	//3次元復元結果読み込み
	void loadReconstructFile(const std::string& filename);

	//コーナー検出によるプロジェクタ位置姿勢を推定
	bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat projframe, 
														cv::Mat initialR, cv::Mat initialT, 
														cv::Mat &dstR, cv::Mat &dstT, 
														cv::Mat &error,
														cv::Mat &dstR_predict, cv::Mat &dstT_predict,
														int dotsCount, int dots_data[],
														double thresh, 
														bool isKalman, bool isPredict,
													   /*cv::Mat &draw_camimage,*/ cv::Mat &draw_projimage);

	//処理時間計測・DebugLog表示用
	void startTic()
	{
		cTimeStart = CFileTime::GetCurrentTime();// 現在時刻
	}

	//処理時間計測用・DebugLog表示用
	//文字列が長すぎると文字化ける
	void stopTic(std::string label);

	//csvファイルから円の座標を読み込む
	bool ProjectorEstimation::loadDots(std::vector<cv::Point2f> &corners, cv::Mat &drawimage);


private:
	//計算部分(プロジェクタ点の最近棒を探索する)
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh, bool isKalman, bool isPredict,
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &error, cv::Mat& dstR_predict, cv::Mat& dstT_predict,
												/*cv::Mat &draw_camimage,*/ cv::Mat &chessimage);
	//ランダムにnum点を抽出
	void get_random_points(int num, vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, vector<cv::Point2f>& calib_p, vector<cv::Point3f>& calib_P);

	//対応点からRとTの算出
	int calcParameters(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT);
	//Ceres Solver Version
	int ProjectorEstimation::calcParameters_Ceres(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT);

	//3次元点のプロジェクタ画像への射影と再投影誤差の計算
	void calcReprojectionErrors(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat R, cv::Mat T, vector<cv::Point2d>& projection_P, vector<double>& errors);

	//対応点からRとTの算出(RANSAC)
	int calcParameters_RANSAC(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, int num, float thresh, cv::Mat& dstR, cv::Mat& dstT);

	//回転行列→クォータニオン
	bool transformRotMatToQuaternion(
		double &qx, double &qy, double &qz, double &qw,
		double m11, double m12, double m13,
		double m21, double m22, double m23,
		double m31, double m32, double m33);

};
#endif