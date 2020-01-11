//
// Created by zsk on 18-11-25.
//

#ifndef OF_VELOCITY_SENSOR_H
#define OF_VELOCITY_SENSOR_H
#define GRAVITY (9.81f)
namespace EKFHomography{
    struct IMU {
        float acc[3];
        float gyro[3];
        float timestamp;
    };
}
#endif //OF_VELOCITY_SENSOR_H
