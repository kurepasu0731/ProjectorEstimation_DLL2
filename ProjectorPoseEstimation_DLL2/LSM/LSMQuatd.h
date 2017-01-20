#pragma once

#include "LeastSquaresMethod.h"
#include "LeastSquaresMethodQuadratic.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>



//�N�H�[�^�j�I���̍ŏ����@�N���X
class LSMQuatd
{
protected:
	LeastSquaresMethod *lsm_x, *lsm_y, *lsm_z, *lsm_w;

public:
	//�R���X�g���N�^
	//lsmkind: LSM�̎��
	LSMQuatd(LSMKind lsmkind);
	~LSMQuatd();

	//�Y�p�W����ݒ肷��
	//(0.0��0����ɂȂ�̂Ŕ�����)
	void setForgetFactor(double factor);

	//�V�����f�[�^��ǉ�����
	void addData(double x, glm::quat y);

	//�X���ƐؕЂ𗘗p����y�̒l���v�Z����
	glm::quat calcYValue(double x);

	//�\���f�[�^�����ׂăN���A����
	void clear();


	//////debug
	void debug_getABC(double *a, double *b, double *c);
};