#ifndef WRAPPERFORUNITY_H
#define WRAPPERFORUNITY_H

#pragma once

#include "projectorPoseEstimation.h"

#define DLLExport __declspec (dllexport)

extern "C" {
	//ProjectorEstimationインスタンス生成
	DLLExport void* openProjectorEstimation(int camWidth, int camHeight, int proWidth, int proHeight, double trackingtime, const char* backgroundImgFile, int _checkerRow, int _checkerCol, int _blockSize, int _x_offset, int _y_offset); 

	//パラメータファイル、3次元復元ファイル読み込み
	DLLExport void callloadParam(void* projectorestimation, double initR[], double initT[]);

	//プロジェクタ位置推定コア呼び出し(プロジェクタ画像更新なし)
	DLLExport bool callfindProjectorPose_Corner(void* projectorestimation, /*unsigned char* cam_data,*/
																	int dotsCount, int dots_data[],
																	double _initR[], double _initT[], double _dstR[], double _dstT[], double aveError[],
																	//int camCornerNum, double camMinDist, int projCornerNum, double projMinDist, 
																	double thresh, 
																	int mode, 
																	bool isKalman, bool isPredict);
																	//double C, int dotsMin, int dotsMax, float resizeScale);

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