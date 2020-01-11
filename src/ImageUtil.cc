#include "ImageUtil.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace EKFHomography{

    void ConvertImageToFloat(Mat & image)
    {
        //image.convertTo(image, CV_32F);
        double min,max, average;

        minMaxLoc(image,&min,&max);
        //printf("min, max: %f %f\n", min, max);
        const float v = 1.0/(max - min);
        //image.convertTo(image, CV_32F);
        image.convertTo(image, CV_32F, v, -min * v);
        //std::cout << image << std::endl;
        //float *img = (float*)image.data;
        //printf("image: %f %f %f\n", img[0], img[1], img[2]);
        assert(image.isContinuous());
    }

    void ComputeImageDerivatives(const Mat & image, Mat & imageDx, Mat &imageDy)
    {
        int ddepth = -1; //same image depth as source
        double scale = 1/32.0;// normalize wrt scharr mask for having exact gradient
        double delta = 0;

        Scharr(image, imageDx, ddepth, 1, 0, scale, delta, BORDER_REFLECT );
        Scharr(image, imageDy, ddepth, 0, 1, scale, delta, BORDER_REFLECT );
    }

    Mat SmoothImage(const float sigma, const Mat &im)
    {
        Mat smoothedImage;
        int s = max(5, 2*int(sigma)+1);
        Size kernelSize(s, s);
        GaussianBlur(im, smoothedImage, kernelSize, sigma, sigma, BORDER_REFLECT);
        return smoothedImage;
    }

    cv::Mat skew(const cv::Mat &a)
    {
        cv::Mat ax(cv::Mat::zeros(3,3,CV_32FC1));
        float x = a.at<float>(0);
        float y = a.at<float>(1);
        float z = a.at<float>(2);
        ax.at<float>(0, 1) = -z;
        ax.at<float>(0, 2) = y;
        ax.at<float>(1, 0) = z;
        ax.at<float>(1, 2) = -x;
        ax.at<float>(2, 0) = -y;
        ax.at<float>(2, 1) = x;
        return ax;
    }

    cv::Mat exp(const cv::Mat &theta){
        float phi = cv::norm(theta);
        if(phi<1e-3)
            return cv::Mat::eye(3,3,CV_32FC1);
        cv::Mat phiv = theta/phi;
        cv::Mat phix = skew(phiv);
        cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1) + sin(phi)*phix+(1-cos(phi))*phix*phix;
        return R;
    }
}

