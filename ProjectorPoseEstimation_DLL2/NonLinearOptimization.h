#ifndef NONLINEAROPTIMIZATION_H
#define NONLINEAROPTIMIZATION_H

#pragma once

#include <opencv2\opencv.hpp>

#include <Eigen/Dense>
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

using namespace Eigen;
using namespace std;

	// Generic functor
	template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
	struct Functor
	{
	  typedef _Scalar Scalar;
	  enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	  };
	  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
	  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
	};

	struct misra1a_functor : Functor<double>
	{
		// 目的関数
		misra1a_functor(int inputs, int values, std::vector<cv::Point2f>& proj_p, std::vector<cv::Point3f>& world_p, const cv::Mat& proj_K)
			: inputs_(inputs),
			  values_(values), 
			  proj_p_(proj_p),
			  worldPoints_(world_p),
			  projK(proj_K){}
			  //cam_p_(cam_p), 
			  //reconstructPoints_(reconstructPoints),
			  //cam_K_(cam_K), 
			  //proj_K_(proj_K),
			  //projK_inv_t(proj_K_.inv().t()), 
			  //camK_inv(cam_K.inv()) {}
    
		vector<cv::Point2f> proj_p_;
		vector<cv::Point3f> worldPoints_;
		const cv::Mat projK;

		//**エピポーラ方程式を用いた最適化**//

		//Rの自由度3にする
		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	//回転ベクトルから回転行列にする
		//	Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
		//	Mat R(3, 3, CV_64F, Scalar::all(0));
		//	Rodrigues(rotateVec, R);
		//	//[t]x
		//	Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);
		//	for (int i = 0; i < values_; ++i) {
		//		Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
		//		Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);
		//		Mat error = pp.t() * projK_inv_t * tx * R * camK_inv * cp;
		//		fvec[i] = error.at<double>(0, 0);
		//	}
		//	return 0;
		//}


		//**3次元復元結果を用いた最適化**//

		//Rの自由度3
		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{
			cv::Mat vr = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			cv::Mat vt = (cv::Mat_<double>(3, 1) << _Rt[3], _Rt[4], _Rt[5]);
			cv::Mat R_33(3, 3, CV_64F, cv::Scalar::all(0));
			Rodrigues(vr, R_33);

			// 2次元(プロジェクタ画像)平面へ投影
			std::vector<cv::Point2f> pt;
			cv::projectPoints(worldPoints_, R_33, vt, projK, cv::Mat(), pt);

			// 射影誤差算出
			for (int i = 0; i < values_; ++i) 
			{
				//cv::Mat wp = (cv::Mat_<double>(4, 1) << worldPoints_[i].x, worldPoints_[i].y, worldPoints_[i].z, 1);
				////4*4行列にする
				//cv::Mat Rt = (cv::Mat_<double>(4, 4) << R_33.at<double>(0,0), R_33.at<double>(0,1), R_33.at<double>(0,2), _Rt[3],
				//	                               R_33.at<double>(1,0), R_33.at<double>(1,1), R_33.at<double>(1,2), _Rt[4],
				//								   R_33.at<double>(2,0), R_33.at<double>(2,1), R_33.at<double>(2,2), _Rt[5],
				//								   0, 0, 0, 1);
				//cv::Mat dst_p = projK * Rt * wp;
				//cv::Point2f project_p(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				//// 射影誤差算出
				//fvec[i] = pow(project_p.x - proj_p_[i].x, 2) + pow(project_p.y - proj_p_[i].y, 2);
				// 射影誤差算出
				fvec[i] = pow(pt[i].x - proj_p_[i].x, 2) + pow(pt[i].y - proj_p_[i].y, 2);
				//std::cout << "fvec[" << i << "]: " << fvec[i] << std::endl;
			}
			return 0;
		}

		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

	};

#endif