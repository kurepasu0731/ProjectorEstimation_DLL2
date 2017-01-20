#pragma once

#include "LeastSquaresMethod.h"
#include "LeastSquaresMethodQuadratic.h"
#include <opencv2/opencv.hpp>



//Point3f�̍ŏ����@�N���X
class LSMPoint3f
{
protected:
	LeastSquaresMethod *lsm_x, *lsm_y, *lsm_z;

public:
	//�R���X�g���N�^
	//lsmkind: LSM�̎��
	LSMPoint3f(LSMKind lsmkind);
	~LSMPoint3f();

	//�Y�p�W����ݒ肷��
	//(0.0��0����ɂȂ�̂Ŕ�����)
	void setForgetFactor(double factor);

	//�V�����f�[�^��ǉ�����
	void addData(double x, cv::Point3f y);

	//�X���ƐؕЂ𗘗p����y�̒l���v�Z����
	cv::Point3f calcYValue(double x);

	//�\���f�[�^�����ׂăN���A����
	void clear();

	//////debug
	void debug_getABC(double *a, double *b, double *c);
};