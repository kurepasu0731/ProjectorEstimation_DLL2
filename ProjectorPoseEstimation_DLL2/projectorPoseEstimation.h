#ifndef PROJECTOPOSERESTIMATION_H
#define PROJECTORPOSEESTIMATION_H

#pragma once

#include "WebCamera.h"
#include "NonLinearOptimization.h"

#include <opencv2/opencv.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>


#define DLLExport __declspec (dllexport)


//using namespace cv;
using namespace std;

class ProjectorEstimation
{
public:
	WebCamera* camera;
	WebCamera* projector;

	std::vector<cv::Point2f> projectorImageCorners; //プロジェクタ画像上の対応点座標
	std::vector<cv::Point2f> cameraImageCorners; //カメラ画像上の対応点座標

	//カメラ画像上のコーナー点
	std::vector<cv::Point2f> camcorners;
	//プロジェクタ画像上のコーナー点
	std::vector<cv::Point2f> projcorners;

	//3次元点(カメラ中心)LookUpテーブル
	//** index = カメラ画素(左上始まり)
	//** int image_x = i % CAMERA_WIDTH;
	//** int image_y = (int)(i / CAMERA_WIDTH);
	//** Point3f = カメラ画素の3次元座標(計測されていない場合は(-1, -1, -1))
	std::vector<cv::Point3f> reconstructPoints;

	//プロジェクタ画像
	cv::Mat proj_img, proj_drawimg;


	//コンストラクタ
	ProjectorEstimation(int camwidth, int camheight, int prowidth, int proheight)
	{
		camera = new WebCamera(camwidth, camheight);
		projector = new WebCamera(prowidth, proheight);

		//プロジェクタ画像読み込み,描画用画像作成
		proj_img = cv::imread("Assets/Image/bedsidemusic_1280_800.jpg");
		proj_drawimg =  proj_img.clone();
		cv::undistort(proj_img, proj_drawimg, projector->cam_K, projector->cam_dist);
		
	};

	~ProjectorEstimation(){};

	//キャリブレーションファイル読み込み
	void loadProCamCalibFile(const std::string& filename);

	//3次元復元結果読み込み
	void loadReconstructFile(const std::string& filename);

	//コーナー検出によるプロジェクタ位置姿勢を推定
	bool findProjectorPose_Corner(const cv::Mat& camframe, const cv::Mat projframe, cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, 
		int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode, cv::Mat &draw_camimage, cv::Mat &draw_projimage);

private:
	//計算部分(プロジェクタ点の最近棒を探索する)
	int calcProjectorPose_Corner1(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//計算部分(カメラ点(3次元点)の最近傍を探索する)
	int calcProjectorPose_Corner2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, 
												cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage);

	//コーナー検出
	bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, double minDistance, double num, cv::Mat &drawimage);

	//各対応点の重心位置を計算
	void calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, 
									cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve);
};

extern "C" {
	//ProjectorEstimationインスタンス生成
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight); 

	//パラメータファイル、3次元復元ファイル読み込み
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//プロジェクタ位置推定コア呼び出し
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, unsigned char* cam_data, 
																	double initR[], double initT[], double dstR[], double dstT[],
																	int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, int mode);
	//ウィンドウ破棄
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};
}
#endif