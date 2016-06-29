#ifndef WRAPPERFORUNITY_H
#define WRAPPERFORUNITY_H

#pragma once

#include "projectorPoseEstimation.h"


#define DLLExport __declspec (dllexport)

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
#endif