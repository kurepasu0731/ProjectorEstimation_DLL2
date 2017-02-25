#pragma once

#include <opencv2/opencv.hpp>

#define DLLExport __declspec (dllexport)


using namespace cv;


extern "C" {
	DLLExport void loadCameraParam(double projectionMatrix[], double dist[]);

	DLLExport void loadProjectorParam(double projectionMatrix[], double dist[]);

	DLLExport void  loadExternalParam(double R[], double T[]);

}