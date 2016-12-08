#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"
#include "KalmanFilter.h"

#include <opencv2/opencv.hpp>
#include <random>

//PCL
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>

#include <atltime.h>

#include<fstream>
#include<iostream>
#include<string>
#include<sstream> //文字ストリーム



//#define DLLExport __declspec (dllexport)


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;
	cv::Size checkerPattern;

	std::vector<cv::Point2f> projectorImageCorners; //プロジェクタ画像上の対応点座標(チェッカパターンによる推定の場合)
	std::vector<cv::Point2f> cameraImageCorners; //カメラ画像上の対応点座標

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


	//過去30？フレーム分の対応点間距離
	std::vector<std::vector<double>> preDistsArrays;
	int preframesize;
	int sum;

	//処理時間計測
	CFileTime cTimeStart, cTimeEnd;
	CFileTimeSpan cTimeSpan;

	//--フラグ関係--//
	bool detect_proj; //プロジェクタ画像のコーナー点を検出したかどうか

	//コンストラクタ
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);
		checkerPattern = cv::Size(_checkerCol, _checkerRow);

		kf.initKalmanfilter(6, 3, 0, 1);//等速度
		//kf.initKalmanfilter(9, 3, 0, 1);//等加速度

		//プロジェクタ画像読み込み,描画用画像作成
		proj_img = cv::imread(backgroundImgFile);
		proj_undist =  proj_img.clone();
		cv::undistort(proj_img, proj_undist, projector->cam_K, projector->cam_dist);

		//チェッカパターンによる推定の場合
		//プロジェクタ画像上の交点座標を求めておく
		getProjectorImageCorners(projectorImageCorners, _checkerRow, _checkerCol, _blockSize, _x_offset, _y_offset);

		//1フレーム前の対応点間距離の初期化
		for(int i = 0; i < projectorImageCorners.size(); i++)
		{
			preDists.emplace_back(0.0);
		}

		std::vector<double> array;
		array.clear();
		for(int i = 0; i < projectorImageCorners.size(); i++)
		{
			preDistsArrays.emplace_back(array);
		}
		preframesize = 20;
		//sum計算
		sum = 0;
		for(int i = 0; i < preframesize; i++)
			sum += i;

		//コーナー検出の場合
		//TODO:プロジェクタ画像上のコーナー点を求めておく
		detect_proj = false;
	};

	~ProjectorEstimation(){};

	//キャリブレーションファイル読み込み
	void loadProCamCalibFile(const std::string& filename);

	//3次元復元結果読み込み
	void loadReconstructFile(const std::string& filename);

	//コーナー検出によるプロジェクタ位置姿勢を推定
	bool ProjectorEstimation::findProjectorPose_Corner(const cv::Mat projframe, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &error,
													   int dotsCount, int dots_data[],
													   //int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, 
													   double thresh, 
													   int mode,
													   bool isKalman,
													   //double C, int dotsMin, int dotsMax, float resizeScale,
													   cv::Mat &draw_camimage, cv::Mat &draw_projimage);

	//チェッカボード検出によるプロジェクタ位置姿勢を推定
	bool findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage);

	//コーナー検出
	bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage);

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
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, double thresh, bool isKalman,
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &error, cv::Mat &draw_camimage, cv::Mat &chessimage);

	//計算部分(カメラ点(3次元点)の最近傍を探索する)
	int calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &draw_camimage, cv::Mat &chessimage);

	////コーナー検出
	//bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage);

	//各対応点の重心位置を計算
	void calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
									cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve);
	
	//チェッカパターンによる推定の場合

	//計算部分(Rの自由度3)
	int calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//カメラ画像をチェッカパターン検出する
	bool getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image);

	//プロジェクタ画像上の交点座標を求める
	void getProjectorImageCorners(std::vector<cv::Point2f>& projPoint, int _row, int _col, int _blockSize, int _x_offset, int _y_offset);

	//回転行列→クォータニオン
	bool transformRotMatToQuaternion(
		double &qx, double &qy, double &qz, double &qw,
		double m11, double m12, double m13,
		double m21, double m22, double m23,
		double m31, double m32, double m33);

	//ランダムにnum点を抽出
	void get_random_points(int num, vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, vector<cv::Point2f>& calib_p, vector<cv::Point3f>& calib_P);

	//対応点からRとTの算出
	int calcParameters(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT);

	//3次元点のプロジェクタ画像への射影と再投影誤差の計算
	void calcReprojectionErrors(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat R, cv::Mat T, vector<cv::Point2d>& projection_P, vector<double>& errors);

	//対応点からRとTの算出(RANSAC)
	int calcParameters_RANSAC(vector<cv::Point2f> src_p, vector<cv::Point3f> src_P, cv::Mat initialR, cv::Mat initialT, int num, float thresh, cv::Mat& dstR, cv::Mat& dstT);

	//**ランダムドットマーカー用**//	
	//ドット検出
	bool ProjectorEstimation::getDots(cv::Mat &src, std::vector<cv::Point2f> &corners, double C, int dots_thresh_min, int dots_thresh_max, float resizeScale, cv::Mat &drawimage);

	void ProjectorEstimation::calCoG_dot_v0(cv::Mat &src, cv::Point& sum, int& cnt, cv::Point& min, cv::Point& max, cv::Point p);

};

/*
extern "C" {
	//ProjectorEstimationインスタンス生成
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset, double _thresh); 

	//パラメータファイル、3次元復元ファイル読み込み
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//プロジェクタ位置推定コア呼び出し
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, unsigned char* prj_data, 
																	double initR[], double initT[], double dstR[], double dstT[],
																	int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode);
	//ウィンドウ破棄
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};

	//カメラ画像用マスクの作成
	DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data);

}
*/
#endif