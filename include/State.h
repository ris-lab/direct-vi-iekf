//
// Created by zsk on 18-11-25.
//

#ifndef OF_VELOCITY_STATE_H
#define OF_VELOCITY_STATE_H

#include "Sensor.h"
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "NormalVectorElement.h"
#include <glog/logging.h>
#include "ParameterConfig.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/video.hpp>
#include <algorithm>
#include "ImageUtil.h"

#define START_CV_TIME(time) double time = cv::getTickCount()
#define END_CV_TIME_MS(time) time = (cv::getTickCount() - time)/cv::getTickFrequency()*1e3
#define LOG_END_CV_TIME_MS(time) LOG(INFO) << "[Time Cost] " << #time << " = "<< (cv::getTickCount() - time)/cv::getTickFrequency()*1e3 << " ms"

namespace EKFHomography{
    template <typename Scalar, int D, int DQ>
    class State{
    public:
        typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
        typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
        typedef NormalVectorElement<Scalar> NormalVector;

        typedef Eigen::Matrix<Scalar, D, D> MatrixState;
        typedef Eigen::Matrix<Scalar, D, 1> VectorState;
        typedef Eigen::Matrix<Scalar, DQ, DQ> MatrixIMUNoise;
        typedef MatrixState MatrixStateNoise;
        static const int stateDim = D;
        static const int IMUNoiseDim = DQ;

        State(Scalar alpha, const Vector3 &v, const Vector2 &normal, const Vector2 &gravity0,
              const MatrixIMUNoise &Qimu, const MatrixState &P0, const MatrixStateNoise& stateNoise);

        // n is parametrized using the method in rovio
        // assuming that the gravitational vector is aligned with the plane normal vector
        void PropagateIMU(const std::vector<IMU> &imu);
        void PropagateIMU(const IMU &imu);

        // lk method for 6 states(n = [nx ny]')
        void MeasurementUpdateLK(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt);

        //dense direct method for 14 or 8 states including gravity direction
        void MeasurementUpdateDirect(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt);

        Vector3 Velocity() const {
            return mRatioV/mAlpha;
        }

        Scalar Distance2Plane() const {
            return 1./mAlpha;
        }

        Vector3 NormalVec(){
            return mUnitDirection.getVec();
        }

        Vector3 GravityVec(){
            return mGravityDirection.getVec();
        }

        Eigen::Array<Scalar, 10, 1> Sigma(){
            Eigen::Matrix<Scalar, 10, 8> J_d_v_g_v2error;
            Scalar d = 1./mAlpha;
            J_d_v_g_v2error.setZero();
            J_d_v_g_v2error(0, 0) = -d*d;
            J_d_v_g_v2error.template block<3,1>(1, 0) = -d*d*mRatioV;
            J_d_v_g_v2error.template block<3,3>(1, 1) = d*Eigen::Matrix<Scalar, 3, 3>::Identity();
            J_d_v_g_v2error.template block<3,2>(4, 4) = mUnitDirection.getM();
            J_d_v_g_v2error.template block<3,2>(7, 6) = mGravityDirection.getM();
            Eigen::Matrix<Scalar, 10, 10> sigmam = J_d_v_g_v2error*mP.template block<8, 8>(0, 0)*J_d_v_g_v2error.transpose();
            Eigen::Array<Scalar, 10, 1> sigmav = sigmam.diagonal().array().sqrt();
            return sigmav;
        }

    protected:
        Vector3 mRatioV, mBa, mBw;
        Vector2 mNormal;
        NormalVector mUnitDirection, mGravityDirection;

        Scalar mAlpha, mCurrentTime;
        MatrixState mP;
        MatrixIMUNoise mQimu;
        Matrix3 mK, mKinv;
        MatrixStateNoise mStateNoise;
        cv::Mat mCVK, mCVKinv, mCVDist, remap1l_, remap2l_;
        cv::Mat Ixk_1, Iyk_1, imk_1;
        bool mbInit, mbHuber, mbLogDebug;
    };

    #include "State.hpp"
    typedef State<double, 14, 6> StateD14;
}

#endif //OF_VELOCITY_STATE_H
