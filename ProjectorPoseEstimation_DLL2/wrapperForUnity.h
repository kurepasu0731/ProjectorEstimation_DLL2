#ifndef WRAPPERFORUNITY_H
#define WRAPPERFORUNITY_H

#pragma once

#include "projectorPoseEstimation.h"

#define DLLExport __declspec (dllexport)

extern "C" {
	//ProjectorEstimationインスタンス生成
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, double trackingtime, const char* backgroundImgFile); 

	//パラメータファイル、3次元復元ファイル読み込み
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//プロジェクタ位置推定コア呼び出し(プロジェクタ画像更新なし)
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, /*unsigned char* cam_data,*/
																	int dotsCount, int dots_data[],
																	double _initR[], double _initT[], 
																	double _dstR[], double _dstT[], 
																	double aveError[],
																	double _dstR_predict[], double _dstT_predict[],
																	double thresh, 
																	bool isKalman, bool isPredict);

	//ウィンドウ破棄
	DLLExport void destroyAllWindows()
	{
		cv::destroyAllWindows();
	};

	//カメラ画像用マスクの作成
	DLLExport void createCameraMask(void* projectorestimation, unsigned char* cam_data);


	//ドット確認用
	DLLExport void checkDotsArray(void* projectorestimation, unsigned char* cam_data, int dotsCount, int dots_data[]);
    
}
#endif