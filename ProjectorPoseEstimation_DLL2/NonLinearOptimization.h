#ifndef NONLINEAROPTIMIZATION_H
#define NONLINEAROPTIMIZATION_H

#pragma once

#include <opencv2\opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry> //Eigen��Geometry�֘A�̊֐����g���ꍇ�C���ꂪ�K�v
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
		// �ړI�֐�
		misra1a_functor(int inputs, int values, std::vector<cv::Point2f>& proj_p, std::vector<cv::Point3f>& world_p, const cv::Mat& _projK_34)
			: inputs_(inputs),
			  values_(values), 
			  proj_p_(proj_p),
			  worldPoints_(world_p),
			  projK_34(_projK_34){}
    
		vector<cv::Point2f> proj_p_;
		vector<cv::Point3f> worldPoints_;
		vector<double> weight;
		const cv::Mat projK_34;

		//**3�����������ʂ�p�����œK��**//

		//R�̎��R�x3(���P�F�N�H�[�^�j�I��)
		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{
			//��]
			// Compute w from the unit quaternion(��]�Ɋւ���N�H�[�^�j�I���̃m������1)
			Quaterniond q(0, _Rt[0], _Rt[1], _Rt[2]);
			q.w () = static_cast<double> (sqrt (1 - q.dot (q)));
			q.normalize ();
			MatrixXd qMat = q.toRotationMatrix();

			//��]�s��
			cv::Mat R_33 = (cv::Mat_<double>(3, 3) << qMat(0, 0), qMat(0, 1), qMat(0, 2), qMat(1, 0), qMat(1, 1), qMat(1, 2), qMat(2, 0), qMat(2, 1), qMat(2, 2));
			//���i�x�N�g��
			cv::Mat vt = (cv::Mat_<double>(3, 1) << _Rt[3], _Rt[4], _Rt[5]);
			//4*4�s��ɂ���
			cv::Mat Rt = (cv::Mat_<double>(4, 4) << R_33.at<double>(0,0), R_33.at<double>(0,1), R_33.at<double>(0,2), vt.at<double>(0, 0),
					                            R_33.at<double>(1,0), R_33.at<double>(1,1), R_33.at<double>(1,2), vt.at<double>(1, 0),
												R_33.at<double>(2,0), R_33.at<double>(2,1), R_33.at<double>(2,2), vt.at<double>(2, 0),
												0, 0, 0, 1);
			// �ˉe�덷�Z�o
			for (int i = 0; i < values_; ++i) 
			{
				// 2����(�v���W�F�N�^�摜)���ʂ֓��e
				cv::Mat wp = (cv::Mat_<double>(4, 1) << worldPoints_[i].x, worldPoints_[i].y, worldPoints_[i].z, 1);
				cv::Mat dst_p = projK_34 * Rt * wp;
				cv::Point2d project_p(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				// �ˉe�덷�Z�o(�d�݂�����)
				fvec[i] = sqrt(pow(project_p.x - proj_p_[i].x, 2) + pow(project_p.y - proj_p_[i].y, 2));
			}
			return 0;
		}

		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

	};

#endif