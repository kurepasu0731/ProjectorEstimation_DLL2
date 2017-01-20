#pragma once

#include <stdlib.h>
#include <algorithm>


//�Y�p�W���t���ŏ����@�N���X (1���֐��ߎ�)
class LeastSquaresMethod
{
protected:
	double n;
	double sigma_x;
	double sigma_y;
	double sigma_x_square;
	double sigma_xy;
	double a;
	double b;
	double ff;

public:
	LeastSquaresMethod();
	virtual ~LeastSquaresMethod();

	//�Y�p�W����ݒ肷��
	//(0.0��0����ɂȂ�̂Ŕ�����)
	virtual void setForgetFactor(double factor);

	//�V�����f�[�^(x, y)��ǉ�����
	virtual void addData(double x, double y);

	//�����̌X���ƐؕЂ��擾����
	virtual void getAB(double *dst_a, double *dst_b);

	//�X���ƐؕЂ𗘗p����y�̒l���v�Z����
	virtual double calcYValue(double x);

	//�\���f�[�^���N���A����
	virtual void clear();
};


class LeastSquaresMethodSmoothChange : public LeastSquaresMethod
{
protected:
	double a_before;
	double b_before;
	double x_before;
	double x_smoothend;
	double x_latest_calcY;
	double inter_a;
	double inter_b;

public:
	LeastSquaresMethodSmoothChange();
	virtual ~LeastSquaresMethodSmoothChange();

	virtual void addData(double x, double y);
	virtual void getAB(double *dst_a, double *dst_b);
	virtual double calcYValue(double x);
	virtual void clear();
};