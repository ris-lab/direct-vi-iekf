#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include "Sensor.h"
namespace EKFHomography {

    class VIDataReader {
    public:

        VIDataReader(const std::string &dataPath, double timeShift);

        ~VIDataReader();


        bool GetImageIMU(unsigned char *imgBuffer,
                float *imuData,
                double &timestamp,
                float &fTime,
                std::vector<IMU> &imuRawData
        );

    protected:

        bool GetRawImgIMU(std::vector<IMU> &imuRawData,
                          double curTimeStamp,
                          float &fTimeStamp);

        IMU IMUFromVec(const std::vector<float> &raw);

        double GetTimestampFromImgName(const std::string &imgName, int64_t &realTime);

        std::string mImgPath;
        std::ifstream mifIMURawFile;
        double mdTimeScale, mdTimeShift;
        cv::Mat mFrame, mGray;
        std::vector<float> mLastIMURawData;
        double mdStartTime;
        bool mbInitTime;
        std::vector<std::string> mImageNames;
        int mnImg;
        bool mbInit;
    };

}