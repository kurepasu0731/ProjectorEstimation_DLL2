#include <ceres/ceres.h>
#include <ceres/rotation.h>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


// Converts a given Euler angles to Rotation Matrix
template<typename T>
void euler2rot(const T* const euler,  T rotationMatrix[9])
{
	T x = euler[0];
	T y = euler[1];
	T z = euler[2];

	// Assuming the angles are in radians.
	T ch = cos(z);
	T sh = sin(z);
	T ca = cos(y);
	T sa = sin(y);
	T cb = cos(x);
	T sb = sin(x);

	T m00, m01, m02, m10, m11, m12, m20, m21, m22;

	m00 = ch * ca;
	m01 = sh*sb - ch*sa*cb;
	m02 = ch*sa*sb + sh*cb;
	m10 = sa;
	m11 = ca*cb;
	m12 = -ca*sb;
	m20 = -sh*ca;
	m21 = sh*sa*cb + ch*sb;
	m22 = -sh*sa*sb + ch*cb;

	rotationMatrix[0] = m00;
	rotationMatrix[1] = m01;
	rotationMatrix[2] = m02;
	rotationMatrix[3] = m10;
	rotationMatrix[4] = m11;
	rotationMatrix[5] = m12;
	rotationMatrix[6] = m20;
	rotationMatrix[7] = m21;
	rotationMatrix[8] = m22;
}


struct ErrorFunction{
  ErrorFunction(cv::Point2d ip, cv::Point3d wp, const cv::Mat& _projK_34)
	  :	projK_34(_projK_34){
    p2d[0] = ip.x; p2d[1] = ip.y;
    p3d[0] = wp.x; p3d[1] = wp.y; p3d[2] = wp.z;
  }
  template<typename T>
  bool operator () (const T* const Rt, T* residual) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3], wpT[3];
	wpT[0] = T(p3d[0]); wpT[1] = T(p3d[1]); wpT[2] = T(p3d[2]);

	//// Compute w from the unit quaternion(âÒì]Ç…ä÷Ç∑ÇÈÉNÉHÅ[É^ÉjÉIÉìÇÃÉmÉãÉÄÇÕ1)
	//T w = sqrt(T(1.0) - Rt[0] * Rt[0] + Rt[1] * Rt[1] + Rt[2] * Rt[2]);
	//T R_quat[4] = {Rt[0], Rt[1], Rt[2], w};
	//T R_matrix[9];
	//ceres::QuaternionRotatePoint(R_quat, wpT, p);
	//p[0] += Rt[3];
	//p[1] += Rt[4];
	//p[2] += Rt[5];

	//âÒì]
	T R_matrix[9];
	T R_euler[3] = {Rt[0], Rt[1], Rt[2]};
	euler2rot(R_euler, R_matrix);

	//pÇÃà⁄ìÆ
	p[0] = R_matrix[0] * wpT[0] + R_matrix[1] * wpT[1] + R_matrix[2] * wpT[2] + Rt[3];
	p[1] = R_matrix[3] * wpT[0] + R_matrix[4] * wpT[1] + R_matrix[5] * wpT[2] + Rt[4];
	p[2] = R_matrix[6] * wpT[0] + R_matrix[7] * wpT[1] + R_matrix[8] * wpT[2] + Rt[5];

	//éÀâe
	T P_00 = T(projK_34.at<double>(0,0));
	T P_01 = T(projK_34.at<double>(0,1));
	T P_02 = T(projK_34.at<double>(0,2));
	T P_03 = T(projK_34.at<double>(0,3));

	T P_10 = T(projK_34.at<double>(1,0));
	T P_11 = T(projK_34.at<double>(1,1));
	T P_12 = T(projK_34.at<double>(1,2));
	T P_13 = T(projK_34.at<double>(1,3));

	T P_20 = T(projK_34.at<double>(2,0));
	T P_21 = T(projK_34.at<double>(2,1));
	T P_22 = T(projK_34.at<double>(2,2));
	T P_23 = T(projK_34.at<double>(2,3));

    //éÀâeçsóÒÇópÇ¢Çƒìäâe
	T projected[3];
	projected[0] = P_00 * p[0] + P_01 * p[1] + P_02 * p[2] + P_03;
	projected[1] = P_10 * p[0] + P_11 * p[1] + P_12 * p[2] + P_13;
	projected[2] = P_20 * p[0] + P_21 * p[1] + P_22 * p[2] + P_23;

    //äœë™ç¿ïWÇ∆ìäâeç¿ïWÇÃç∑
    residual[0] = projected[0]/projected[2] - p2d[0];
    residual[1] = projected[1]/projected[2] - p2d[1];


    return true;
  }
  double p2d[2];
  double p3d[3];
  const cv::Mat projK_34;
};