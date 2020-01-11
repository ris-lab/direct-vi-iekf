//
// Created by Zhong,Shangkun on 2018/9/20.
//

#include "ParameterConfig.h"
#include <glog/logging.h>
#include <iostream>
int g_origin_frame_cols = 1280;
int g_origin_frame_rows = 720;
int g_frame_cols = 640;
int g_frame_rows = 480;
float g_fx = 500.f;
float g_fy = 500.f;
float g_cx = 300.f;
float g_cy = 200.f;
float g_k1 = 0.f;
float g_k2 = 0.f;
float g_p1 = 0.f;
float g_p2 = 0.f;
float g_new_fx = 450.f;
float g_new_fy = 450.f;
float g_new_cx = 320.f;
float g_new_cy = 220.f;
float g_scale = 1.f;
float g_im_weight = 1.f;
float g_sigma_az = 1.f;
float g_sigma_ay = 1.f;
float g_sigma_ax = 1.f;

float g_sigma_ba = 1.f;
float g_sigma_wx = 1.f;
float g_sigma_wy = 1.f;
float g_sigma_wz = 1.f;
float g_sigma_bw = 1.f;

int g_max_iteration = 3;
int g_use_lk = 0;

int g_use_Tci = 1;
cv::Mat g_Tci = cv::Mat::eye(4,4,CV_32FC1);
cv::Mat g_Tic = cv::Mat::eye(4,4,CV_32FC1);
int g_level = 0;
double g_time_shift = 0;

float g_sigma_alpha;
float g_sigma_vartheta_xy;
float g_sigma_vartheta_z;
float g_sigma_n;
float g_sigma_g;

float g_sigma_alpha0;
float g_sigma_vartheta0;
float g_sigma_n0;
float g_sigma_g0;
float g_sigma_ba0;
float g_sigma_bw0;
float g_robust_delta = 0.2;
int g_show_log;
int g_use_huber;

float g_alpha0 = 10.f;
float g_vartheta0x = 0.f;
float g_vartheta0y = 0.f;
float g_vartheta0z = 0.f;
float g_n0x = 0.f;
float g_n0y = 0.f;
float g_g0x = 0.f;
float g_g0y = 0.f;
void ReadGlobalParaFromYaml(const std::string &paraDir){
    cv::FileStorage fsSettings(paraDir.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << paraDir << std::endl;
        exit(-1);
    }
    fsSettings["Config.FrameRows"] >> g_frame_rows;
    fsSettings["Config.FrameCols"] >> g_frame_cols;
    fsSettings["Config.OriginFrameRows"] >> g_origin_frame_rows;
    fsSettings["Config.OriginFrameCols"] >> g_origin_frame_cols;
    fsSettings["Config.DownSample"] >> g_level;
    fsSettings["Config.Scale"] >> g_scale;
    fsSettings["Config.ImageWeight"] >> g_im_weight;
    fsSettings["Config.MaxIteration"] >> g_max_iteration;
    fsSettings["Config.UseLK"] >> g_use_lk;
    fsSettings["Config.SigmaAx"] >> g_sigma_ax;
    fsSettings["Config.SigmaAy"] >> g_sigma_ay;
    fsSettings["Config.SigmaAz"] >> g_sigma_az;
    fsSettings["Config.SigmaBa"] >> g_sigma_ba;
    fsSettings["Config.SigmaWx"] >> g_sigma_wx;
    fsSettings["Config.SigmaWy"] >> g_sigma_wy;
    fsSettings["Config.SigmaWz"] >> g_sigma_wz;
    fsSettings["Config.SigmaBw"] >> g_sigma_bw;
    fsSettings["Config.SigmaAlpha"] >> g_sigma_alpha;
    fsSettings["Config.SigmaVarthetaXY"] >> g_sigma_vartheta_xy;
    fsSettings["Config.SigmaVarthetaZ"] >> g_sigma_vartheta_z;
    fsSettings["Config.SigmaN"] >> g_sigma_n;
    fsSettings["Config.SigmaG"] >> g_sigma_g;
    fsSettings["Config.UseHuber"] >> g_use_huber;

    fsSettings["Config.SigmaAlpha0"] >> g_sigma_alpha0;
    fsSettings["Config.SigmaVartheta0"] >> g_sigma_vartheta0;
    fsSettings["Config.SigmaN0"] >> g_sigma_n0;
    fsSettings["Config.SigmaG0"] >> g_sigma_g0;
    fsSettings["Config.SigmaBa0"] >> g_sigma_ba0;
    fsSettings["Config.SigmaBw0"] >> g_sigma_bw0;
    fsSettings["Config.RobustDelta"] >> g_robust_delta;

    fsSettings["Camera.fx"] >> g_fx;
    fsSettings["Camera.fy"] >> g_fy;
    fsSettings["Camera.cx"] >> g_cx;
    fsSettings["Camera.cy"] >> g_cy;

    fsSettings["Camera.k1"] >> g_k1;
    fsSettings["Camera.k2"] >> g_k2;
    fsSettings["Camera.p1"] >> g_p1;
    fsSettings["Camera.p2"] >> g_p2;

    fsSettings["Camera.newfx"] >> g_new_fx;
    fsSettings["Camera.newfy"] >> g_new_fy;
    fsSettings["Camera.newcx"] >> g_new_cx;
    fsSettings["Camera.newcy"] >> g_new_cy;

    fsSettings["CameraIMU.bTci"] >> g_use_Tci;
    fsSettings["Config.ShowLog"] >> g_show_log;

    fsSettings["Init.Alpha0"] >> g_alpha0;
    fsSettings["Init.Vartheta0X"] >> g_vartheta0x;
    fsSettings["Init.Vartheta0Y"] >> g_vartheta0y;
    fsSettings["Init.Vartheta0Z"] >> g_vartheta0z;
    fsSettings["Init.N0X"] >> g_n0x;
    fsSettings["Init.N0Y"] >> g_n0y;
    fsSettings["Init.G0X"] >> g_g0x;
    fsSettings["Init.G0Y"] >> g_g0y;



	if(g_use_Tci > 0){
		cv::Mat Tci;
		fsSettings["CameraIMU.T"] >> Tci;
		g_Tci = cv::Mat::eye(4, 4, CV_32FC1);
		Tci.copyTo(g_Tci.rowRange(0, 3));
		g_Tic.rowRange(0, 3).colRange(0, 3) = g_Tci.rowRange(0, 3).colRange(0, 3).t();
		g_Tic.rowRange(0, 3).col(3) = -g_Tic.rowRange(0, 3).colRange(0, 3)*g_Tci.rowRange(0, 3).col(3);
	}
	else{
		cv::Mat Tic;
		fsSettings["CameraIMU.T"] >> Tic;
		g_Tic = cv::Mat::eye(4, 4, CV_32FC1);
		Tic.copyTo(g_Tic.rowRange(0, 3));
		g_Tci.rowRange(0, 3).colRange(0, 3) = g_Tic.rowRange(0, 3).colRange(0, 3).t();
		g_Tci.rowRange(0, 3).col(3) = -g_Tci.rowRange(0, 3).colRange(0, 3)*g_Tic.rowRange(0, 3).col(3);
	}
    fsSettings["CameraIMU.TimeShift"] >> g_time_shift;
    LOG(INFO) << "- fx: " << g_fx;
    LOG(INFO) << "- fy: " << g_fy;
    LOG(INFO) << "- cx: " << g_cx;
    LOG(INFO) << "- cy: " << g_cy;
    LOG(INFO) << "- time shift: " << g_time_shift;
    LOG(INFO) << "- k1: " << g_k1;
    LOG(INFO) << "- k2: " << g_k2;
    LOG(INFO) << "- p1: " << g_p1;
    LOG(INFO) << "- p2: " << g_p2;
    LOG(INFO) << "- new fx: " << g_new_fx;
    LOG(INFO) << "- new fy: " << g_new_fy;
    LOG(INFO) << "- new cx: " << g_new_cx;
    LOG(INFO) << "- new cy: " << g_new_cy;
    LOG(INFO) << "- sigma_ax: " << g_sigma_ax;
    LOG(INFO) << "- sigma_ay: " << g_sigma_ay;
    LOG(INFO) << "- sigma_az: " << g_sigma_az;
    LOG(INFO) << "- sigma_ba: " << g_sigma_ba;
    LOG(INFO) << "- sigma_wx: " << g_sigma_wx;
    LOG(INFO) << "- sigma_wy: " << g_sigma_wy;
    LOG(INFO) << "- sigma_wz: " << g_sigma_wz;
    LOG(INFO) << "- sigma_bw: " << g_sigma_bw;
    LOG(INFO) << "- sigma_alpha: " << g_sigma_alpha;
    LOG(INFO) << "- sigma_n: " << g_sigma_n;
    LOG(INFO) << "- sigma_g: " << g_sigma_g;
    LOG(INFO) << "- sigma_vartheta_xy: " << g_sigma_vartheta_xy;
    LOG(INFO) << "- singma_vartheta_z: " << g_sigma_vartheta_z;
    LOG(INFO) << "- sigma_alpha0: " << g_sigma_alpha0;
    LOG(INFO) << "- sigma_n0: " << g_sigma_n0;
    LOG(INFO) << "- sigma_g0: " << g_sigma_g0;
    LOG(INFO) << "- sigma_vartheta0: " << g_sigma_vartheta0;
    LOG(INFO) << "- singma_ba0: " << g_sigma_ba0;
    LOG(INFO) << "- singma_bw0: " << g_sigma_bw0;
    LOG(INFO) << "- alpha0: " << g_alpha0;
    LOG(INFO) << "- vartheta0x: " << g_vartheta0x;
    LOG(INFO) << "- vartheta0y: " << g_vartheta0y;
    LOG(INFO) << "- vartheta0z: " << g_vartheta0z;
    LOG(INFO) << "- n0x: " << g_n0x;
    LOG(INFO) << "- n0y: " << g_n0y;
    LOG(INFO) << "- g0x: " << g_g0x;
    LOG(INFO) << "- g0y: " << g_g0y;

}
