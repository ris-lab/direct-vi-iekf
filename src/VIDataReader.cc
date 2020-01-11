//
// Created by zsk on 1/4/20.
//
#include "VIDataReader.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <algorithm>

namespace EKFHomography{

    VIDataReader::VIDataReader(const std::string &dataPath, double timeShift) :
            mbInit(false), mnImg(0), mbInitTime(false), mdTimeShift(timeShift) {

        //std::string imu_raw_path = data_path + "/imu_rot.csv";
        std::string imuRawPath = dataPath + "/imu0.csv";
        LOG(INFO) << "imu_raw_path: " << imuRawPath;
        mifIMURawFile.open(imuRawPath);
        if (!mifIMURawFile.is_open()) {
            LOG(FATAL) << "imu file does not exist" << std::endl;
            return;
        }
        mImgPath = dataPath + "/left";

        if (!boost::filesystem::is_directory(mImgPath)) {
            LOG(ERROR) << " Check the image path";
            return;
        }
        mdTimeScale = 1e-6;

        std::vector<boost::filesystem::path> imgFileNames;  // so we can sort them later
        std::copy(boost::filesystem::directory_iterator(mImgPath), boost::filesystem::directory_iterator(),
                  std::back_inserter(imgFileNames));
        //std::sort(img_file_names.begin(), img_file_names.end());

        for (auto t : imgFileNames) {
            //LOG(INFO) << t.string();
            std::string str = t.string();
            int idx = str.rfind('.') + 1;
            std::string ext = str.substr(idx, str.length() - idx);
            //LOG(INFO)<<ext;
            if (ext != "png" && ext != "jpg")
                continue;
            //LOG(INFO) << str;
            mImageNames.push_back(str);
        }

        std::sort(mImageNames.begin(), mImageNames.end(), [](const std::string &a, std::string &b) {
            int aidx1 = a.rfind('/') + 1;
            int aidx2 = a.rfind('.');
            uint64_t name1 = atoll(a.substr(aidx1, aidx2 - aidx1).c_str());

            int bidx1 = b.rfind('/') + 1;
            int bidx2 = b.rfind('.');
            uint64_t name2 = atoll(b.substr(bidx1, bidx2 - bidx1).c_str());
            return name1 <= name2;
        });

        LOG(INFO) << "load img size:" << mImageNames.size();
    }

    VIDataReader::~VIDataReader() {
        if (mifIMURawFile.is_open()) {
            mifIMURawFile.close();
        }

    };


    bool VIDataReader::GetImageIMU(unsigned char *imgBuffer, float *imuData, double &timestamp,
                                   float &fTime,
                                   std::vector<IMU> &imuRawData
    ) {
        timestamp = 0;
        double curTimeStamp = 0;

        if (mnImg >= mImageNames.size())
            return false;
        // discard img data before first imu
        std::string tmpstr;
        if (!mbInit) {
            do {
                getline(mifIMURawFile, tmpstr, '\n');
            } while (!(tmpstr[0] >= '0' && tmpstr[0] < '9') && tmpstr[0] != '.');

            int pos = tmpstr.find(',');
            double first_imu_t;
            first_imu_t = atof(tmpstr.substr(0, pos).c_str()) * mdTimeScale;
            do {
//                LOG(INFO) << mImageNames[mnImg];
                const std::string &impath = mImageNames[mnImg];
				LOG(INFO) << "im path: " << impath;
                mFrame = cv::imread(impath);
                int64_t realTime;
                curTimeStamp = GetTimestampFromImgName(impath, realTime);
                curTimeStamp = curTimeStamp + mdTimeShift;
                if (mFrame.empty() || mnImg >= mImageNames.size())
                    return false;
                mnImg++;
                if (curTimeStamp > first_imu_t + 0.03) {
                    LOG(INFO) << "cur time: " << curTimeStamp;
                    mbInit = true;
                    LOG(INFO) << "set true";
                }
                //else
                //{
                //	LOG(INFO) << "jump this image at start";
                //}
            } while (!mbInit);
        } else {
            if (mFrame.empty() || mnImg >= mImageNames.size())
                return false;
            const std::string &impath = mImageNames[mnImg];
            mFrame = cv::imread(impath);
            LOG(INFO) << "img path: " << impath;
            int64_t realTime;
            curTimeStamp = GetTimestampFromImgName(impath, realTime);
            curTimeStamp = curTimeStamp + mdTimeShift;
            //mnImg+=2;
            mnImg++;
        }


        if (mFrame.channels() == 3)
#if CV_VERSION_MAJOR == 4
            cv::cvtColor(mFrame, mGray, cv::COLOR_BGR2GRAY);
#else
            cv::cvtColor(mFrame, mGray, CV_BGR2GRAY);
#endif
        else if (mFrame.channels() == 1)
            mGray = mFrame;

        memcpy(imgBuffer, (unsigned char *) mGray.data,
               sizeof(unsigned char) * mGray.rows * mGray.cols);

        bool imueof = GetRawImgIMU(imuRawData, curTimeStamp, fTime);

        timestamp = curTimeStamp;

        return imueof;
    }

    double VIDataReader::GetTimestampFromImgName(const std::string &imgName, int64_t &realTime) {
        std::string timeStr = boost::filesystem::path(imgName).stem().string();
        int64_t t = boost::lexical_cast<uint64_t>(timeStr);
        realTime = t;
        return double(t) * mdTimeScale;
    }

    bool VIDataReader::GetRawImgIMU(std::vector<IMU> &imuRawData,
                                    double curTimeStamp,
                                    float &fTimeStamp) {

        if (!mLastIMURawData.empty()) {
            if (mLastIMURawData[0] + mdStartTime > curTimeStamp) {
                //printf("%f\n", mLastIMURawData[0]);
                LOG(INFO) << "last imu time > cur image time";
                return true;
            }

            imuRawData.push_back(IMUFromVec(mLastIMURawData));
            mLastIMURawData.clear();
        }


        const int jump = 1;
        int lines = 1;
        while (true) {
            if (mifIMURawFile.eof()) {
//                LOG(FATAL) << "imu raw data eof";
                return false;
            }

            std::string tmp;
            std::vector<float> tmpdata;
            getline(mifIMURawFile, tmp, '\n');
            //LOG(INFO) << tmp;
            if (tmp.empty())
                break;
            std::stringstream ss(tmp);
            std::string token;
            char delim = ',';

            std::getline(ss, token, delim);
            double timestamp = atof(token.c_str()) * mdTimeScale;
            if (!mbInitTime) {
                mdStartTime = curTimeStamp;
                mbInitTime = true;
            }
            float time2 = timestamp - mdStartTime;
            tmpdata.push_back(time2);

            while (std::getline(ss, token, delim)) {
                tmpdata.push_back(atof(token.c_str()));
//                LOG(INFO) << tmpdata.back();
            }
//            LOG(INFO) << "imu raw timestamp: " << std::fixed << std::setprecision(6) << timestamp;


            //if (cur_time_stamp <= timestamp - 0.0002 || cur_time_stamp < timestamp + 0.0002) {
            if (curTimeStamp <= timestamp) {
                mLastIMURawData.assign(tmpdata.begin(), tmpdata.end());
                break;
            }
//            LOG(INFO)
            imuRawData.push_back(IMUFromVec(tmpdata));
            //tmpdata.clear();
        }

        fTimeStamp = curTimeStamp - mdStartTime;
        return true;
    }

    IMU VIDataReader::IMUFromVec(const std::vector<float> &raw) {
        IMU imu;
        imu.timestamp = raw[0];
        std::copy_n(&raw[1], 3, imu.acc);
        std::copy_n(&raw[4], 3, imu.gyro);
        //std::copy_n(&raw[7], 9, imu.rot);
        return imu;
    }
}
