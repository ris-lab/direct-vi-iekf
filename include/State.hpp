//
// Created by zsk on 18-11-25.
//


//#define DRAW_OPTICAL_FLOW

template<typename Scalar, int _D, int _DQ>
State<Scalar, _D, _DQ>::State(Scalar alpha, const Vector3 &v, const Vector2 &normal, const Vector2 &gravity0,
                              const MatrixIMUNoise &Qimu, const MatrixState &P0, const MatrixState &stateNoise):
        mAlpha(alpha), mRatioV(v), mNormal(normal), mUnitDirection(
        Vector3(mNormal(0), mNormal(1), sqrt(1 - mNormal(0) * mNormal(0) - mNormal(1) * mNormal(1)))),
        mGravityDirection(Vector3(gravity0(0),
                                  gravity0(1), sqrt(1 - gravity0(0) * gravity0(0) - gravity0(1) * gravity0(1)))),
        mBa(0, 0, 0), mBw(0, 0, 0),
        mP(P0),
        mStateNoise(stateNoise),
        mQimu(Qimu), mCurrentTime(0.f),
        mbInit(false),
        mbHuber(g_use_huber),
        mbLogDebug(g_show_log > 0) {
    mK.setIdentity();
//    const float s = 1.f/(1<<g_level);
    const float s = g_scale;

    mCVK = cv::Mat::eye(3, 3, CV_32FC1);
    mCVDist = (cv::Mat_<float>(4, 1) << g_k1, g_k2, g_p1, g_p2);
    mCVK.at<float>(0, 0) = g_fx * s;
    mCVK.at<float>(1, 1) = g_fy * s;
    mCVK.at<float>(0, 2) = g_cx * s;
    mCVK.at<float>(1, 2) = g_cy * s;
    mCVKinv = mCVK.inv();

    cv::Mat mNewK;
    if (g_k1 != 0) {
        //mNewK = cv::getOptimalNewCameraMatrix(mCVK, mCVDist, cv::Size(g_frame_cols*s, g_frame_rows*s), 0);
        mNewK = cv::Mat::eye(3, 3, CV_32FC1);
        mNewK.at<float>(0, 0) = g_new_fx * s;
        mNewK.at<float>(1, 1) = g_new_fy * s;
        mNewK.at<float>(0, 2) = g_new_cx * s;
        mNewK.at<float>(1, 2) = g_new_cy * s;
        LOG(INFO) << "new K: " << mNewK;
        cv::initUndistortRectifyMap(mCVK, mCVDist, cv::Mat(), mNewK,
                                    cv::Size(g_frame_cols * s * 0.8, g_frame_rows * s * 0.8), CV_16SC2, remap1l_,
                                    remap2l_);
    }

    if (!mNewK.empty()) {
        mK(0, 0) = mNewK.at<float>(0, 0);
        mK(1, 1) = mNewK.at<float>(1, 1);
        mK(0, 2) = mNewK.at<float>(0, 2);
        mK(1, 2) = mNewK.at<float>(1, 2);
        mKinv = mK.inverse();
    } else {
        mK(0, 0) = g_fx * s;
        mK(1, 1) = g_fy * s;
        mK(0, 2) = g_cx * s;
        mK(1, 2) = g_cy * s;
        mKinv = mK.inverse();
    }
}

template<typename Scalar, int D, int _DQ>
void State<Scalar, D, _DQ>::PropagateIMU(const std::vector<IMU> &imus) {
    for (const auto &imu:imus)
        PropagateIMU(imu);
}

template<typename Scalar, int _D, int _DQ>
void State<Scalar, _D, _DQ>::PropagateIMU(const IMU &imu) {
    if (!mbInit) {
        mCurrentTime = imu.timestamp;
        mbInit = true;
    }
    assert(imu.timestamp >= mCurrentTime);
    Scalar dt = imu.timestamp - mCurrentTime;
    const Scalar g0 = -GRAVITY;
    Vector3 acc, gyro;

    for (int i = 0; i < 3; i++) {
        acc(i) = imu.acc[i] - mBa(i);
        gyro(i) = imu.gyro[i] - mBw(i);
    }

    Vector3 g = mGravityDirection.getVec();
    if (mbLogDebug)
        LOG(INFO) << "dt : " << dt << " ori acc: " << acc.transpose() << " acc: " <<
                  (acc - g0 * g).transpose() << " gyro: " << gyro.transpose() << " n: "
                  << mUnitDirection.getVec().transpose()
                  << " g: " << g.transpose();

    Matrix3 skewGyro = skew(gyro);
    Vector3 n = mUnitDirection.getVec();

    Scalar nTv = n.dot(mRatioV);
    Scalar alpha = (1 + nTv * dt) * mAlpha;
    Vector3 ratioV = mRatioV + dt * mAlpha * (acc - g0 * g) -
                     dt * skewGyro * mRatioV + dt * nTv * mRatioV;

    Matrix3 I3;
    I3.setIdentity();

    typename NormalVector::V3D dm = -dt * (I3 - n * n.transpose()) * gyro;
    typename NormalVector::QPD qm = qm.exponentialMap(dm);
    NormalVector nOut = mUnitDirection.rotated(qm);

    typename NormalVector::V3D dmg = -dt * (I3 - g * g.transpose()) * gyro;
    typename NormalVector::QPD qmg = qm.exponentialMap(dmg);
    NormalVector gOut = mGravityDirection.rotated(qmg);

    // error propagatoin
    MatrixState F;

    F.setIdentity();
    Eigen::Matrix<Scalar, 3, 2> M = mUnitDirection.getM();
    Eigen::Matrix<Scalar, 3, 2> Mg = mGravityDirection.getM();
    Eigen::Matrix<Scalar, 2, 3> G23 =
            nOut.getM().transpose() * NormalVector::gSM(qm.rotate(n)) * NormalVector::Lmat(dm) *
            (I3 - n * n.transpose());
    Eigen::Matrix<Scalar, 2, 3> G23g =
            gOut.getM().transpose() * NormalVector::gSM(qmg.rotate(n)) * NormalVector::Lmat(dmg) *
            (I3 - g * g.transpose());
    Matrix3 skewV = skew(mRatioV);

    //jac of altitude
    F(0, 0) += nTv * dt;
    F.template block<1, 3>(0, 1) = mAlpha * n.transpose() * dt;
    F.template block<1, 2>(0, 4) = mAlpha * mRatioV.transpose() * M * dt;

    // jac of translational optic flow
    F.template block<3, 1>(1, 0) = (acc - g0 * g) * dt;
    F.template block<3, 3>(1, 1) += (-skewGyro + nTv * I3 + mRatioV * n.transpose()) * dt;
    F.template block<3, 2>(1, 4) = mRatioV * mRatioV.transpose() * M * dt;
    F.template block<3, 2>(1, 6) = -g0 * mAlpha * Mg * dt;
    if (_D > 8) {
        F.template block<3, 3>(1, 8) = -mAlpha * I3 * dt;
        F.template block<3, 3>(1, 11) = -skewV * dt;
    }

    // jac of normal vector
    Vector3 dw = -gyro * dt;
    F.template block<2, 2>(4, 4) = nOut.getM().transpose() * (
            NormalVector::gSM(qm.rotate(n)) * NormalVector::Lmat(dm) * (
                    -(I3 * n.dot(dw) + n * dw.transpose()))
            + typename NormalVector::MPD(qm).matrix()
    ) * M;
//    LOG(INFO) << "Lmat: " << NormalVector::Lmat(dm);

    // jac of gravity vector
    F.template block<2, 2>(6, 6) = gOut.getM().transpose() * (
            NormalVector::gSM(qmg.rotate(g)) * NormalVector::Lmat(dmg) * (
                    -(I3 * g.dot(dw) + g * dw.transpose()))
            + typename NormalVector::MPD(qmg).matrix()
    ) * Mg;
//    LOG(INFO) << "Lmat: " << NormalVector::Lmat(dmg);

    if (_D > 8) {
        F.template block<2, 3>(4, 11) = G23 * dt;
        F.template block<2, 3>(6, 11) = G23g * dt;
    }
    if (mbLogDebug)
        LOG(INFO) << "F = " << F;

    Scalar sqrtime = sqrt(dt);
    Eigen::Matrix<Scalar, _D, _DQ> G;
    G.setZero();
    G.template block<3, 3>(1, 0) = -mAlpha * I3 * sqrtime;
    G.template block<3, 3>(1, 3) = -skewV * sqrtime;
    G.template block<2, 3>(4, 3) = G23 * sqrtime;
    G.template block<2, 3>(6, 3) = G23g * sqrtime;

    if (mbLogDebug)
        LOG(INFO) << "G = " << G;

    Eigen::Matrix<Scalar, _D, _D> G2;
    G2.setZero();
    G2(0, 0) = sqrtime;
    G2(1, 1) = sqrtime;
    G2(2, 2) = sqrtime;
    G2(3, 3) = sqrtime;
    G2.template block<2, 2>(4, 4) = G23 * mUnitDirection.getN() * sqrtime;
    G2.template block<2, 2>(6, 6) = G23g * mGravityDirection.getN() * sqrtime;
    if (_D > 8) {
        G2.template block<3, 3>(8, 8) = I3 * sqrtime;
        G2.template block<3, 3>(11, 11) = I3 * sqrtime;
    }
    if (mbLogDebug)
        LOG(INFO) << "G2 = " << G2;

    mP = F * mP * F.transpose() + G * mQimu * G.transpose() + G2 * mStateNoise * G2.transpose();
    mP = (mP + mP.transpose()) / 2.;
    if (mbLogDebug)
        LOG(INFO) << "mP = " << mP;
    mAlpha = alpha;
    mRatioV = ratioV;
    mUnitDirection.q_ = nOut.q_;
    mGravityDirection.q_ = gOut.q_;
    mCurrentTime = imu.timestamp;
    if (mbLogDebug)
        LOG(INFO) << "alpha = " << mAlpha << " ratioV = " << mRatioV.transpose() << " mNormal = "
                  << nOut.getVec().transpose();
}

template<typename Scalar, int _D, int _DQ>
void State<Scalar, _D, _DQ>::MeasurementUpdateLK(const std::vector<IMU> &imus, const cv::Mat &imLast,
                                                 const cv::Mat &imCur, Scalar dt) {
    Vector3 gyro(0, 0, 0);
    for (const IMU &imu:imus) {
        gyro(0) += imu.gyro[0];
        gyro(1) += imu.gyro[1];
        gyro(2) += imu.gyro[2];
        if (mbLogDebug)
            LOG(INFO) << "gyro: " << imu.gyro[0] << " " << imu.gyro[1] << " " << imu.gyro[2];
    }
    gyro = 1.f / imus.size() * gyro;
    if (mbLogDebug) {
        LOG(INFO) << "gyro = " << gyro.transpose();

        LOG(INFO) << "K = " << mCVK << " Kinv = " << mCVDist;
    }

    cv::Mat imk;
//
//	if(g_scale == 1.f) {
//		imk = imCur.clone();
//		imk_1 = imLast.clone();
//	}
//	else
    {

        if (imk_1.empty()) {
            cv::resize(imLast, imk_1, cv::Size(), g_scale, g_scale);
            cv::remap(imk_1, imk_1, remap1l_, remap2l_, cv::INTER_LINEAR);
        }
        cv::resize(imCur, imk, cv::Size(), g_scale, g_scale);
        cv::remap(imk, imk, remap1l_, remap2l_, cv::INTER_LINEAR);
    }
    std::vector<cv::Point2f> kpsk, kpsk_1, kpskun, kpsk_1un;
    cv::goodFeaturesToTrack(imk, kpsk, 50, 0.1, 10);
    std::vector<uchar> status;
    std::vector<float> err;
    int levels = 3, win = 20;
    cv::calcOpticalFlowPyrLK(imk, imk_1, kpsk, kpsk_1, status, err, cv::Size(win, win), levels);
//#define DRAW_MATCHES
#ifdef DRAW_MATCHES
    cv::Mat imMatches = drawMatches(imk, imk_1, kpsk, kpsk_1, status);
    static int nimg = 0;
    char buf[256];
    sprintf(buf, "/tmp/of_match_%06d.png", nimg++);
    cv::imwrite(buf, imMatches);
#endif
    kpskun = kpsk;
    kpsk_1un = kpsk_1;
    LOG(INFO) << "#good features: " << kpsk.size();
//    cv::imshow("Ik_1", imLast);
//    if (mbLogDebug){
//
//    }
    if (mbLogDebug) {
        cv::imshow("Ik", imCur);
        cv::waitKey(1);
    }


    Vector3 ez(0, 0, 1);
    Matrix3 I3;
    I3.setIdentity();

    int iter = 0;
    int maxIter = g_max_iteration;

    float th2 = g_robust_delta * g_robust_delta;
    MatrixState Pinv0 = mP.inverse();
    const float &fx = mK(0, 0);
    const float &fy = mK(1, 1);
    const float &cx = mK(0, 2);
    const float &cy = mK(1, 2);
    Scalar alpha0 = mAlpha;
    Vector3 ba0 = mBa;
    Vector3 bw0 = mBw;
    Vector3 ratioV0 = mRatioV;
    NormalVector normal0, gravity0;
    normal0.q_ = mUnitDirection.q_;
    gravity0.q_ = mGravityDirection.q_;
    while (true) {
        Vector3 n, gyro_b;
        gyro_b = gyro - mBw;
        n = mUnitDirection.getVec();
        const Scalar &nx = n(0);
        const Scalar &ny = n(1);
        const Scalar &nz = n(2);
        const Scalar &vx = mRatioV(0);
        const Scalar &vy = mRatioV(1);
        const Scalar &vz = mRatioV(2);
//        LOG(INFO) << n.transpose();

        Matrix3 H = mK * (skew(gyro_b) + mRatioV * n.transpose()) * mKinv;
        MatrixState imHessian;
        imHessian.setZero();
        VectorState JTe;
        JTe.setZero();
        Eigen::Matrix<Scalar, 3, 2> N = mUnitDirection.getM();

        Scalar chi2 = 0.f;
        int nmatches = 0;
        std::vector<cv::Point2f> predictedPoints;
        for (int i = 0, iend = kpsk.size(); i < iend; i++) {
            const cv::Point2f &pt = kpskun[i];
            nmatches++;
            float x = pt.x;
            float y = pt.y;

            Vector3 p(x, y, 1);
            Matrix3 I3_pezT = (I3 - p * ez.transpose());
            Vector3 pk_1 = p + dt * I3_pezT * H * p;
            predictedPoints.push_back(cv::Point2f(pk_1(0), pk_1(1)));
            if (!status[i]) continue;

            Eigen::Matrix<Scalar, 2, _D> J2;
            J2.setZero();
            Eigen::Matrix<Scalar, 2, 3> Jn;

            Jn(0, 0) = -(cx / fx - x / fx) * (vz * (cx * dt - dt * x) + dt * fx * vx);
            Jn(0, 1) = -(cy / fy - y / fy) * (vz * (cx * dt - dt * x) + dt * fx * vx);
            Jn(0, 2) = vz * (cx * dt - dt * x) + dt * fx * vx;
            Jn(1, 0) = -(cx / fx - x / fx) * (vz * (cy * dt - dt * y) + dt * fy * vy);
            Jn(1, 1) = -(cy / fy - y / fy) * (vz * (cy * dt - dt * y) + dt * fy * vy);
            Jn(1, 2) = vz * (cy * dt - dt * y) + dt * fy * vy;

            J2(0, 1) = -dt * fx * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
            J2(0, 3) = dt * x * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy)) -
                       cx * dt * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
            J2(1, 2) = -dt * fy * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
            J2(1, 3) = dt * y * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy)) -
                       cy * dt * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
            J2.template block<2, 2>(0, 4) = Jn * N;

            if (_D > 8) {
                J2(0, 11) = (cy / fy - y / fy) * (cx * dt - dt * x);
                J2(0, 12) = -dt * fx - (cx / fx - x / fx) * (cx * dt - dt * x);
                J2(0, 13) = -dt * fx * (cy / fy - y / fy);
                J2(1, 11) = dt * fy + (cy / fy - y / fy) * (cy * dt - dt * y);
                J2(1, 12) = -(cx / fx - x / fx) * (cy * dt - dt * y);
                J2(1, 13) = dt * fy * (cx / fx - x / fx);
            }

            Scalar w = 1;
            Vector2 pk_1mea(kpsk_1un[i].x, kpsk_1un[i].y);
            Vector2 res = (pk_1mea - pk_1.template head<2>());

            Scalar cur_chi2 = res.dot(res);
            Scalar res_norm = sqrt(cur_chi2);

            if (mbHuber) {
                if (fabs(res_norm) > g_robust_delta)
                    w *= g_robust_delta / fabs(res_norm);
            }
            imHessian += J2.transpose() * J2 * w;

            JTe += J2.transpose() * res * w;
            chi2 += cur_chi2;
        }

//        cv::Mat imMatches2 = drawMatches(imk, imk_1, kpsk, predictedPoints, status);
//        char buf2[256];
//        sprintf(buf2, "/tmp/of_match_%06d_iter%02d.png", nimg, iter);
//        cv::imwrite(buf2, imMatches2);

        if (mbLogDebug) {
            LOG(INFO) << "nmatches: " << nmatches;
            LOG(INFO) << "H = " << imHessian << std::endl << " JTe = " << JTe.transpose() << " chi2 = " << chi2;
        }

        VectorState xerr;
        xerr(0) = alpha0 - mAlpha;
        xerr.template segment<3>(1) = ratioV0 - mRatioV;
        Vector2 nerr, gerr;
        normal0.boxMinus(mUnitDirection, nerr);
        gravity0.boxMinus(mGravityDirection, gerr);
        if (mbLogDebug) {
            LOG(INFO) << "vector jac: " << mUnitDirection.getRotationFromTwoNormalsJac(mUnitDirection, normal0);
            LOG(INFO) << "N: " << mUnitDirection.getN() << " M: " << normal0.getM();
        }

        xerr.template segment<2>(4) = nerr;
        xerr.template segment<2>(6) = gerr;

        if (_D > 8) {
            xerr.template segment<3>(8) = ba0 - mBa;
            xerr.template segment<3>(11) = bw0 - mBw;
        }

        MatrixState Jprior;
        Jprior.setIdentity();
        typename NormalVector::M2D jboxminus;
        normal0.boxMinusJac(mUnitDirection, jboxminus);
        Jprior.template block<2, 2>(4, 4) = jboxminus;
        gravity0.boxMinusJac(mGravityDirection, jboxminus);
        Jprior.template block<2, 2>(6, 6) = jboxminus;
        MatrixState Pinv;

        Pinv = Jprior.transpose() * Pinv0 * Jprior;
//        LOG(INFO) << "JTe: "
        JTe += Jprior.transpose() * Pinv0 * xerr;

        MatrixState Hessian = (imHessian + Pinv);
        VectorState delta = Hessian.ldlt().solve(JTe);
        if (mbLogDebug) {
            LOG(INFO) << "Pinv: " << Pinv << " Jprior: " << Jprior << " xerr: " << xerr.transpose() << " JT*xerr: "
                      << xerr.transpose() * Pinv.transpose();
            LOG(INFO) << " Hessian = " << Hessian;
            LOG(INFO) << "delta = " << delta.transpose() << " condi = "
                      << Hessian.inverse().norm() * Hessian.norm();
        }
//        if(fabs(delta(0)) < 0.3)
        mAlpha += delta(0);
        if (mAlpha < 0)
            mAlpha *= -1.f;
        mRatioV += delta.template segment<3>(1);
        if (mRatioV.norm() > 0.01 && chi2 < 10 /*&& delta.template segment<2>(4).norm() < 0.2*/) {
            NormalVector nout;
            mUnitDirection.boxPlus(delta.template segment<2>(4), nout);
            if (nout.getVec()(2) > 0)
                mUnitDirection.q_ = nout.q_;
        }
        NormalVector gout;
        mGravityDirection.boxPlus(delta.template segment<2>(6), gout);
        mGravityDirection.q_ = gout.q_;
        if (_D > 8) {
            mBa += delta.template segment<3>(8);
            mBw += delta.template segment<3>(11);
        }
        if (mbLogDebug) {
            LOG(INFO) << "alpha = " << mAlpha << " ratioV: "
                      << mRatioV.transpose() << " normal: " << mNormal.transpose() << " norm(ratioV): "
                      << mRatioV.transpose();
            LOG(INFO) << "ba = " << mBa.transpose() << " bw = " << mBw.transpose() << " norm(delta)= "
                      << delta.norm();
        }


        if (delta.norm() < 0.001 || iter++ >= maxIter) {
            MatrixState A = Hessian.inverse() * imHessian;
            mP = mP - A * mP;
            if (mbLogDebug)
                LOG(INFO) << "poseterior p = " << mP;
            break;
        }
    }
    imk_1 = imk;
}

template<typename Scalar, int _D, int _DQ>
void State<Scalar, _D, _DQ>::MeasurementUpdateDirect(const std::vector<IMU> &imus, const cv::Mat &imLast,
                                                     const cv::Mat &imCur, Scalar dt) {
    Vector3 gyro(0, 0, 0);
    for (const IMU &imu:imus) {
        gyro(0) += imu.gyro[0];
        gyro(1) += imu.gyro[1];
        gyro(2) += imu.gyro[2];
        if (mbLogDebug)
            LOG(INFO) << "gyro: " << imu.gyro[0] << " " << imu.gyro[1] << " " << imu.gyro[2];
    }
    gyro = 1.f / imus.size() * gyro;
    const float s = g_scale;
    if (mbLogDebug) {
        LOG(INFO) << "gyro = " << gyro.transpose();
        LOG(INFO) << "K = " << mK << " Kinv = " << mKinv;
    }
    if (imk_1.empty()) {
        cv::resize(imLast, imk_1, cv::Size(), s, s);
        if (g_k1 != 0) {
            cv::remap(imk_1, imk_1, remap1l_, remap2l_, cv::INTER_LINEAR);
        }
        imk_1 = SmoothImage(3, imk_1);
        ConvertImageToFloat(imk_1);
        ComputeImageDerivatives(imk_1, Ixk_1, Iyk_1);
    }

    cv::Mat imk;
    cv::resize(imCur, imk, cv::Size(), s, s);
    if (g_k1 != 0) {
        cv::remap(imk, imk, remap1l_, remap2l_, cv::INTER_LINEAR);
        //cv::imwrite("undis.png", imk);
    }
    //static int nimg = 0 ;
    //char buf[256];
    //sprintf(buf, "/tmp/%04d.png", nimg++);
    //cv::imwrite(buf, imk);

#ifdef DRAW_OPTICAL_FLOW
    cv::Mat imkBeforeNorm = imk.clone();
#endif
    if (mbLogDebug) {
        cv::imshow("Ik", imCur);
        cv::waitKey(1);
    }


    START_CV_TIME(tSmoothImage);
    imk = SmoothImage(3, imk);
    //double bk = cv::mean(imk);
    //double bk_1 = cv::mean(imk_1);
    //imk_1 += bk_1 - bk;
    LOG_END_CV_TIME_MS(tSmoothImage);
    START_CV_TIME(tConvertImageToFloat);
    ConvertImageToFloat(imk);
    LOG_END_CV_TIME_MS(tConvertImageToFloat);
    cv::Mat Ixk, Iyk;
    START_CV_TIME(tComputeImageDerivatives);
    ComputeImageDerivatives(imk, Ixk, Iyk);
    LOG_END_CV_TIME_MS(tComputeImageDerivatives);

    Vector3 ez(0, 0, 1);
    Matrix3 I3;
    I3.setIdentity();

    int iter = 0;
    int maxIter = g_max_iteration;
    MatrixState Pinv0 = mP.inverse();
    int windowSize = 5;
    int halfWindowSize = windowSize / 2;
//    LOG(INFO) << "PInv0 = " << Pinv0;
    const float *ptrimk_1 = imk_1.ptr<float>(0, 0);
    const float *ptrimk = imk.ptr<float>(0, 0);
    const float *ptrIxk_1 = Ixk_1.ptr<float>(0, 0);
    const float *ptrIyk_1 = Iyk_1.ptr<float>(0, 0);

    const float *ptrIxk = Ixk.ptr<float>(0, 0);
    const float *ptrIyk = Iyk.ptr<float>(0, 0);

    const Scalar &fx = mK(0, 0);
    const Scalar &fy = mK(1, 1);
    const Scalar &cx = mK(0, 2);
    const Scalar &cy = mK(1, 2);
    Scalar alpha0 = mAlpha;
    Vector3 ba0 = mBa;
    Vector3 bw0 = mBw;
    Vector3 ratioV0 = mRatioV;
    NormalVector normal0, gravity0;
    normal0.q_ = mUnitDirection.q_;
    gravity0.q_ = mGravityDirection.q_;
    while (true) {

        START_CV_TIME(tIteration);
        Vector3 n, gyro_b;
        gyro_b = gyro - mBw;
        n = mUnitDirection.getVec();
        const Scalar &nx = n(0);
        const Scalar &ny = n(1);
        const Scalar &nz = n(2);
        const Scalar &vx = mRatioV(0);
        const Scalar &vy = mRatioV(1);
        const Scalar &vz = mRatioV(2);
//        LOG(INFO) << n.transpose();
        Matrix3 H = mK * (skew(gyro_b) + mRatioV * n.transpose()) * mKinv;
        MatrixState imHessian;
        imHessian.setZero();
        VectorState JTe;
        JTe.setZero();
        Eigen::Matrix<Scalar, 3, 2> N = mUnitDirection.getM();

        Scalar chi2 = 0.f;
        int border = 3;
        int step = 1;

        int npixel = 0;
        START_CV_TIME(tHJTe);
        for (int y = border; y < imk.rows - border; y += step)
            for (int x = border; x < imk.cols - border; x += step, npixel++) {

                Vector3 p(x, y, 1);
                Matrix3 I3_pezT = (I3 - p * ez.transpose());
//                Vector3 pk_1 = p + dt * I3_pezT * H * p;
                Vector3 pk_1;
                pk_1(0) = x + x * (dt * H(0, 0) - dt * H(2, 0) * x) + y * (dt * H(0, 1) - dt * H(2, 1) * x) +
                          dt * H(0, 2) - dt * H(2, 2) * x;
                pk_1(1) = y + x * (dt * H(1, 0) - dt * H(2, 0) * y) + y * (dt * H(1, 1) - dt * H(2, 1) * y) +
                          dt * H(1, 2) - dt * H(2, 2) * y;
                if ((pk_1(0)) < 0 || (pk_1(0)) >= imk_1.cols - 1
                    || (pk_1(1)) < 0 || (pk_1(1)) >= imk_1.rows - 1)
                    continue;
                Vector3 Kinvp = mKinv * p;
                Scalar nTKinvp = n.dot(Kinvp);
                Eigen::Matrix<Scalar, 1, 2> J1;
                Eigen::Matrix<Scalar, 2, _D> J2;
                Eigen::Matrix<Scalar, 1, _D> J;
                J2.setZero();
                Eigen::Matrix<Scalar, 2, 3> Jn;


                Jn(0, 0) = -(cx / fx - x / fx) * (vz * (cx * dt - dt * x) + dt * fx * vx);
                Jn(0, 1) = -(cy / fy - y / fy) * (vz * (cx * dt - dt * x) + dt * fx * vx);
                Jn(0, 2) = vz * (cx * dt - dt * x) + dt * fx * vx;
                Jn(1, 0) = -(cx / fx - x / fx) * (vz * (cy * dt - dt * y) + dt * fy * vy);
                Jn(1, 1) = -(cy / fy - y / fy) * (vz * (cy * dt - dt * y) + dt * fy * vy);
                Jn(1, 2) = vz * (cy * dt - dt * y) + dt * fy * vy;

                J2(0, 1) = -dt * fx * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
                J2(0, 3) = dt * x * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy)) -
                           cx * dt * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
                J2(1, 2) = -dt * fy * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
                J2(1, 3) = dt * y * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy)) -
                           cy * dt * (nx * (cx / fx - x / fx) - nz + ny * (cy / fy - y / fy));
                J2.template block<2, 2>(0, 4) = Jn * N;

                if (_D > 8) {
                    J2(0, 11) = (cy / fy - y / fy) * (cx * dt - dt * x);
                    J2(0, 12) = -dt * fx - (cx / fx - x / fx) * (cx * dt - dt * x);
                    J2(0, 13) = -dt * fx * (cy / fy - y / fy);
                    J2(1, 11) = dt * fy + (cy / fy - y / fy) * (cy * dt - dt * y);
                    J2(1, 12) = -(cx / fx - x / fx) * (cy * dt - dt * y);
                    J2(1, 13) = dt * fy * (cx / fx - x / fx);
                }
//                J2.template block<2, 3>(0, 9) = -J2.template block<2, 3>(0, 9);
//                LOG(INFO) << "matlab symbol: " << J2;
//                J2.block<3, 3>(0, 1) = dt*nTKinvp*I3_pezT*mK;
//                J2.block<3, 2>(0, 4) = dt*I3_pezT*mK*mRatioV*Kinvp.transpose()*N;
//                LOG(INFO) << "matrix dot: " << J2;
#define VALUE_FROM_ADDRESS(address) (*(address))
#if 1
                //bilinear interpolation for pixel intensities
                int last_u_i = static_cast<int>(floor(pk_1(0)));
                int last_v_i = static_cast<int>(floor(pk_1(1)));
                const float subpix_u_ref = pk_1(0) - last_u_i;
                const float subpix_v_ref = pk_1(1) - last_v_i;
                const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                const float w_ref_br = subpix_u_ref * subpix_v_ref;
                int cur_u = x;
                int cur_v = y;
                const int step = imk_1.cols;
                int last_idx = last_u_i + last_v_i * step;
                int cur_idx = cur_u + cur_v * step;
                const float *ptr_Ixk_1 = ptrIxk_1 + last_idx;
                float last_gx = w_ref_tl * VALUE_FROM_ADDRESS(ptr_Ixk_1) +
                                w_ref_tr * VALUE_FROM_ADDRESS(ptr_Ixk_1 + 1) +
                                w_ref_bl * VALUE_FROM_ADDRESS(ptr_Ixk_1 + step) +
                                w_ref_br * VALUE_FROM_ADDRESS(ptr_Ixk_1 + step + 1);
                J1(0) = (VALUE_FROM_ADDRESS(ptrIxk + cur_idx) + last_gx) / 2.f;

                const float *ptr_Iyk_1 = ptrIyk_1 + last_idx;
                float last_gy = w_ref_tl * VALUE_FROM_ADDRESS(ptr_Iyk_1) +
                                w_ref_tr * VALUE_FROM_ADDRESS(ptr_Iyk_1 + 1) +
                                w_ref_bl * VALUE_FROM_ADDRESS(ptr_Iyk_1 + step) +
                                w_ref_br * VALUE_FROM_ADDRESS(ptr_Iyk_1 + step + 1);
                J1(1) = (VALUE_FROM_ADDRESS(ptrIyk + cur_idx) + last_gy) / 2.f;
                J = J1 * J2;

                float w = g_im_weight * J1.norm();
//                float w = g_im_weight;

                //LOG(INFO) << "w = " << w;


                const float *ptr_imk_1 = ptrimk_1 + last_idx;
                float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imk_1) +
                                       w_ref_tr * VALUE_FROM_ADDRESS(ptr_imk_1 + 1) +
                                       w_ref_bl * VALUE_FROM_ADDRESS(ptr_imk_1 + step) +
                                       w_ref_br * VALUE_FROM_ADDRESS(ptr_imk_1 + step + 1);
                float res = (VALUE_FROM_ADDRESS(ptrimk + cur_idx) - last_intensity);

                if (mbHuber) {
                    if (fabs(res) > g_robust_delta)
                        w *= g_robust_delta / fabs(res);
                }
                imHessian += J.transpose() * J * w;
#else
                int last_u_i = cvRound(pk_1.at<float>(0));
                int last_v_i = cvRound(pk_1.at<float>(1));
                int cur_u = x;
                int cur_v = y;

                J1.at<float>(0) =(Ixk.at<float>(cur_v, cur_u) + Ixk_1.at<float>(last_v_i, last_u_i)) / 2.f;
                J1.at<float>(1) =(Iyk.at<float>(cur_v, cur_u) + Iyk_1.at<float>(last_v_i, last_u_i)) / 2.f;
                J = J1*J2;
                const float w = 5;
                imHessian += J.t() * J * w;
                float res = (imk.at<float>(cur_v, cur_u) - imk_1.at<float>(last_v_i, last_u_i));
#endif
                JTe += res * J.transpose() * w;
                chi2 += res * res;
            }
        LOG_END_CV_TIME_MS(tHJTe);
        if (mbLogDebug)
            LOG(INFO) << "H = " << imHessian << std::endl << " JTe = " << JTe.transpose()
                      << " chi2 = " << chi2 << " npixel: " << npixel;

        VectorState xerr;
        xerr(0) = alpha0 - mAlpha;
        xerr.template segment<3>(1) = ratioV0 - mRatioV;
        Vector2 nerr, gerr;
        normal0.boxMinus(mUnitDirection, nerr);
        gravity0.boxMinus(mGravityDirection, gerr);
        if (mbLogDebug) {
            LOG(INFO) << "vector jac: " << mUnitDirection.getRotationFromTwoNormalsJac(mUnitDirection, normal0);
            LOG(INFO) << "N: " << mUnitDirection.getN() << " M: " << normal0.getM();
        }

        xerr.template segment<2>(4) = nerr;
        xerr.template segment<2>(6) = gerr;

        if (_D > 8) {
            xerr.template segment<3>(8) = ba0 - mBa;
            xerr.template segment<3>(11) = bw0 - mBw;
        }

        MatrixState Jprior;
        Jprior.setIdentity();
        typename NormalVector::M2D jboxminus;
        normal0.boxMinusJac(mUnitDirection, jboxminus);
        Jprior.template block<2, 2>(4, 4) = jboxminus;
        gravity0.boxMinusJac(mGravityDirection, jboxminus);
        Jprior.template block<2, 2>(6, 6) = jboxminus;
        MatrixState Pinv = Jprior.transpose() * mP * Jprior;
        Pinv = Pinv.inverse();
        JTe += Jprior.transpose() * Pinv0 * xerr;

        MatrixState Hessian = (imHessian + Pinv);
        VectorState delta = Hessian.ldlt().solve(JTe);
        if (mbLogDebug) {
            LOG(INFO) << "Pinv: " << Pinv << " Jprior: " << Jprior << " xerr: " << xerr.transpose() << " JT*xerr: "
                      << xerr.transpose() * Pinv.transpose();
            LOG(INFO) << " Hessian = " << Hessian;
            LOG(INFO) << "delta = " << delta.transpose() << " condi = "
                      << Hessian.inverse().norm() * Hessian.norm();
        }
//        if(fabs(delta(0)) < 0.3)
        mAlpha += delta(0);
        if (mAlpha < 0)
            mAlpha *= -1.f;
        mRatioV += delta.template segment<3>(1);
        //if(mRatioV.norm() > 0.01 && chi2 < 10 /*&& delta.template segment<2>(4).norm() < 0.2*/)
        {
            NormalVector nout;
            mUnitDirection.boxPlus(delta.template segment<2>(4), nout);
            //if (delta.template segment<2>(4).norm() > 0.1){
            //	LOG(INFO) << "change too large";
            //}
            if (nout.getVec()(2) > 0)
                mUnitDirection.q_ = nout.q_;
        }
        NormalVector gout;
        mGravityDirection.boxPlus(delta.template segment<2>(6), gout);
        mGravityDirection.q_ = gout.q_;
        if (_D > 8) {
            mBa += delta.template segment<3>(8);
            mBw += delta.template segment<3>(11);
        }

        if (mbLogDebug) {
            LOG(INFO) << "alpha = " << mAlpha << " ratioV: "
                      << mRatioV.transpose() << " normal: " << mUnitDirection.getVec().transpose()
                      << " norm(ratioV): " << mRatioV.transpose() << " norm(delta_n) = "
                      << delta.template segment<2>(4).norm();
            LOG(INFO) << "ba = " << mBa.transpose() << " bw = " << mBw.transpose() << " norm(delta)= "
                      << delta.norm();
        }

        if (delta.norm() < 5e-2 || iter++ >= maxIter) {
            MatrixState A = Hessian.inverse() * imHessian;
            mP = mP - A * mP;
            mP = (mP + mP.transpose()) / 2.;
            if (mbLogDebug) {
                LOG(INFO) << "poseterior p = " << mP;
            }

#ifdef DRAW_OPTICAL_FLOW
            DrawOpticalFLow(imkBeforeNorm, H, dt);
#endif
            break;
        }
        LOG_END_CV_TIME_MS(tIteration);
    }

    imk_1 = imk;
    Ixk_1 = Ixk;
    Iyk_1 = Iyk;
}
