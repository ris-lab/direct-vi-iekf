//
// Created by zsk on 2018/11/31.
//

#ifndef SETTING_CONFIG_H
#define SETTING_CONFIG_H

#include <string>
#include <opencv2/core/core.hpp>
extern int g_origin_frame_cols;
extern int g_origin_frame_rows;
extern int g_frame_cols;
extern int g_frame_rows;
extern float g_fx;
extern float g_fy;
extern float g_cx;
extern float g_cy;
extern float g_k1;
extern float g_k2;
extern float g_p1;
extern float g_p2;
extern int g_use_Tci;
extern cv::Mat g_Tci;
extern cv::Mat g_Tic;
extern int g_level;
extern float g_scale;
extern double g_time_shift;
extern float g_im_weight;
extern int g_max_iteration;
extern int g_use_lk;
extern float g_sigma_ax;
extern float g_sigma_ay;
extern float g_sigma_az;
extern float g_sigma_ba;
extern float g_sigma_wx;
extern float g_sigma_wy;
extern float g_sigma_wz;
extern float g_sigma_bw;
extern float g_sigma_alpha;
extern float g_sigma_vartheta_xy;
extern float g_sigma_vartheta_z;
extern float g_sigma_g;
extern float g_sigma_n;
extern float g_sigma_alpha0;
extern float g_sigma_vartheta0;
extern float g_sigma_n0;
extern float g_sigma_g0;
extern float g_sigma_ba0;
extern float g_sigma_bw0;
extern float g_robust_delta;
extern int g_show_log;
extern int g_use_huber;
extern float g_new_fx;
extern float g_new_fy;
extern float g_new_cx;
extern float g_new_cy;
extern float g_n0x;
extern float g_n0y;
extern float g_vartheta0x;
extern float g_vartheta0y;
extern float g_vartheta0z;
extern float g_alpha0;
extern float g_g0x;
extern float g_g0y;

void ReadGlobalParaFromYaml(const std::string &paraDir);

#endif //ORB_SLAM2_SETTING_CONFIG_H
