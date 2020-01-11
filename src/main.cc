#include "VIDataReader.h"
#include "State.h"
#include <Eigen/Geometry>
#include <sstream>
#include "Sensor.h"
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>
#include <boost/format.hpp>
#include <iomanip>
using namespace std;

DEFINE_string(yaml_path, "mynt.yaml", "yaml config");
DEFINE_string(data_path, "example_seq/seq0", "video and imu path for test");
DEFINE_string(output_path, ".", "output path for our tests");

void TransformIMU(std::vector<EKFHomography::IMU> &imus, const cv::Mat &Rci, const cv::Mat &cPic){
    for(EKFHomography::IMU& imu:imus)
    {
        cv::Mat acc(3,1,CV_32FC1), gyro(3,1,CV_32FC1);
        for(int j = 0; j<3; j++)
        {
            acc.at<float>(j) = imu.acc[j];
            gyro.at<float>(j) = imu.gyro[j];
        }
        acc = Rci*acc;
        gyro = Rci*gyro;
        cv::Mat skew = (cv::Mat_<float>(3, 3) << 0, -gyro.at<float>(2), gyro.at<float>(1),
                         gyro.at<float>(2), 0, -gyro.at<float>(0),
                        -gyro.at<float>(1), gyro.at<float>(0), 0);
        cv::Mat wwp = skew*skew*cPic;
        LOG(INFO) << "acc rotation compensation: " <<  wwp.t();
        acc += wwp;
        for(int j = 0; j<3; j++)
        {
            imu.acc[j] = acc.at<float>(j);
            imu.gyro[j] = gyro.at<float>(j);
        }
    }
}

typedef EKFHomography::StateD14 StateD;

int main(int argc, char **argv){
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir = "/tmp/";
    FLAGS_alsologtostderr = 0;

    ReadGlobalParaFromYaml(FLAGS_yaml_path);
	//printf("time shift: %f\n", g_time_shift);
    string dataPath = FLAGS_data_path;
    int pos = dataPath.rfind('/')+1;
    string seqName = dataPath.substr(pos, dataPath.length()-pos);
    std::cout << seqName << std::endl;

    EKFHomography::VIDataReader viDataReader(dataPath, g_time_shift);
	unsigned char* buffer = new unsigned char[g_frame_rows*g_frame_cols];
	double timestamp;
	float imuData[12];
	const cv::Mat &Rci = g_Tci.rowRange(0, 3).colRange(0, 3);
	cv::Mat cPic = -g_Tci.rowRange(0, 3).col(3);
	cv::Mat imLast;
	int nimg = 0;

    StateD::Vector3 vartheta0(g_vartheta0x, g_vartheta0y, g_vartheta0z);
    StateD::Vector2 n0(g_n0x, g_n0y);
    StateD::Vector2 gravity0(g_g0x, g_g0y);
    double alpha0 = g_alpha0;

	int nstate = StateD::stateDim;
	int nimuNoise = StateD::IMUNoiseDim;

    StateD::MatrixIMUNoise Qimu;
	Qimu.setIdentity();
	Qimu(0, 0) = g_sigma_ax;
	Qimu(1, 1) = g_sigma_ay;
    Qimu(2, 2) = g_sigma_az;
    Qimu(3, 3) = g_sigma_wx;
    Qimu(4, 4) = g_sigma_wy;
    Qimu(5, 5) = g_sigma_wz;

    StateD::MatrixState P0, stateNoise;
	P0.setIdentity();
	P0(0, 0) = g_sigma_alpha0;
    for(int i = 1; i<4; i++)
        P0(i, i) = g_sigma_vartheta0;
	for(int i = 4; i<6; i++)
	    P0(i, i) = g_sigma_n0;
    for(int i = 6; i<8; i++)
        P0(i, i) = g_sigma_g0;

    stateNoise.setIdentity();
    stateNoise(0, 0) = g_sigma_alpha;
    stateNoise(1, 1) = stateNoise(2, 2) = g_sigma_vartheta_xy;
    stateNoise(3, 3) = g_sigma_vartheta_z;
    stateNoise(4, 4) = stateNoise(5, 5) = g_sigma_n;
    stateNoise(6, 6) = stateNoise(7, 7) = g_sigma_g;

    if (nstate > 8){
        for (int i = 8; i<11; i++)
            P0(i, i) = g_sigma_ba0;
        for(int i = 11; i<nstate; i++)
            P0(i, i) = g_sigma_bw0;
        stateNoise.block<3,3>(8, 8) = StateD::Matrix3::Identity()*g_sigma_ba;
        stateNoise.block<3,3>(11, 11) = StateD::Matrix3::Identity()*g_sigma_bw;
    }


    StateD *state;
    state = new StateD(alpha0, vartheta0, n0, gravity0, Qimu, P0, stateNoise);


	float lastTime;

	char est_buf[256];
	sprintf(est_buf, "%s/%s_est.txt", FLAGS_output_path.c_str(), seqName.c_str());
	ofstream fest(est_buf);
    LOG(INFO) << "Tci: " << g_Tci;
	while(true){
		float fCurTime = 0.0;
		std::vector<EKFHomography::IMU> imuRawData;
		bool b_load = viDataReader.GetImageIMU(buffer, imuData, 
				timestamp, fCurTime, imuRawData);
		if(!b_load) break;
		if(imuRawData.empty()) {
		    LOG(INFO) << "imu empty";
		    continue;
		}
		cv::Mat im(g_frame_rows, g_frame_cols, CV_8UC1, buffer);
		if(nimg++ == 0)
		{
            imLast = im.clone();
			lastTime = fCurTime;
			continue;
		}
		for(int i = 0, iend = imuRawData.size(); i<iend; i++){
            EKFHomography::IMU & imu = imuRawData[i];
			LOG(INFO) << boost::format("%f %f %f %f %f %f %f %f\n") % fCurTime % imu.timestamp % imu.acc[0] % imu.acc[1] % imu.acc[2] % imu.gyro[0] % imu.gyro[1] % imu.gyro[2];
		}
		LOG(INFO) << "nimg: " << nimg << " t = " << fCurTime;
        TransformIMU(imuRawData, Rci, cPic);

        state->PropagateIMU(imuRawData);

		START_CV_TIME(tMeasurementUpdate);
        if(g_use_lk > 0)
            state->MeasurementUpdateLK(imuRawData, imLast, im, fCurTime - lastTime);
        else
            state->MeasurementUpdateDirect(imuRawData, imLast, im, fCurTime - lastTime);

		END_CV_TIME_MS(tMeasurementUpdate);
        float d = state->Distance2Plane();
        StateD::Vector3 estV = state->Velocity();

		float gx, gy, gz;
        StateD::Vector3 gravityDirection = state->GravityVec();
        gx = gravityDirection(0);
        gy = gravityDirection(1);
        gz = gravityDirection(2);

        StateD::Vector3 n = state->NormalVec();
        float nx = n(0);
        float ny = n(1);
        float nz = n(2);
        auto sigmav = state->Sigma();
        float angle = acos(n.dot(gravityDirection));
        fest << std::fixed << std::setprecision(6)
        << timestamp << " " << d << " " << estV(0) << " " << estV(1) << " " << estV(2) 
        << " " << nx << " " << ny << " " << nz <<  " " << gx <<  " " << gy << " " << gz << " " << angle*57.3 << " " << tMeasurementUpdate
        << " " << sigmav(0) << " " << sigmav(1) << " " << sigmav(2) << " " << sigmav(3)
        << " " << sigmav(4) << " " << sigmav(5) << " " << sigmav(6) << " " << sigmav(7)
        << " " << sigmav(8) << " " << sigmav(9)
        << std::endl;
		imLast = im.clone();
		lastTime = fCurTime;
	}
	delete buffer;
	delete state;
    fest.close();

	return 0;
}
