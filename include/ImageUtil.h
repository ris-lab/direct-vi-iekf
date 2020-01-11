#ifndef IMAGE_UTIL_H_
#define IMAGE_UTIL_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
namespace EKFHomography{

    cv::Mat skew(const cv::Mat &a);

    template <typename Derived>
    inline Eigen::Matrix<Derived, 3, 3> skew(const Eigen::Matrix<Derived, 3, 1> &a)
    {
        Eigen::Matrix<Derived, 3, 3> ax;
        ax.setZero();
        Derived x = a(0);
        Derived y = a(1);
        Derived z = a(2);
        ax(0, 1) = -z;
        ax(0, 2) = y;
        ax(1, 0) = z;
        ax(1, 2) = -x;
        ax(2, 0) = -y;
        ax(2, 1) = x;
        return ax;
    }

    // these codes are stolen from Alberto Crivellaro, Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
    //alberto.crivellaro@epfl.ch
    void ConvertImageToFloat(cv::Mat & image);
    cv::Mat SmoothImage(const float sigma, const cv::Mat & im);
    void ComputeImageDerivatives(const cv::Mat & image, cv::Mat & imageDx, cv::Mat &imageDy);
}

#endif
