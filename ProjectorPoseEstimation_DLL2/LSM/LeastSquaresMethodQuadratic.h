#pragma once

#include "LeastSquaresMethod.h"
#include <opencv2/opencv.hpp>


//�Y�p�W���t���ŏ����@�N���X (2���֐��ߎ�)
class LeastSquaresMethodQuadratic : public LeastSquaresMethod
{
protected:
	double sigma_x3;
	double sigma_x4;
	double sigma_x2y;
	double c;


public:
	LeastSquaresMethodQuadratic();
	virtual ~LeastSquaresMethodQuadratic();

	//�V�����f�[�^(x, y)��ǉ�����
	virtual void addData(double x, double y);

	//a��b���擾���� (y = ax^2 + bx + c)
	virtual void getAB(double *dst_a, double *dst_b);

	//c���擾���� (y = ax^2 + bx + c)
	virtual void getC(double *dst_c);

	//�ߎ��Ȑ��𗘗p����y�̒l���v�Z����
	virtual double calcYValue(double x);

	//�\���f�[�^���N���A����
	virtual void clear();
};


/////////debug
//extern bool genelibdebug_global_out_sigmas;
	
class LSMQuadraticSmoothChange : public LeastSquaresMethodQuadratic
{
protected:
	double a_before;
	double b_before;
	double c_before;
	double x_before;
	double x_smoothend;
	double x_latest_calcY;

public:
	LSMQuadraticSmoothChange();
	virtual ~LSMQuadraticSmoothChange();

	virtual void addData(double x, double y);
	virtual double calcYValue(double x);
	virtual void clear();
};


//�ŏ����@�̎��
enum LSMKind
{
	LSM,
	LSM_SmoothChange,
	LSMQuadratic,
	LSMQuadratic_SmoothChange
};