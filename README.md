# Direct  Visual-Inertial  Ego-Motion  Estimation via  Iterated  Extended  Kalman  Filter
By Shangkun Zhong and Pakpong Chirarattananon

## Introduction

This  letter  proposes  a  reactive  navigation  strategyfor recovering the altitude, translational velocity and orientationof  Micro  Aerial  Vehicles.  The  main  contribution  lies  in  the direct  and  tight  fusion  of  Inertial  Measurement  Unit  (IMU) measurements  with  monocular  feedback  under  an  assumption of  a  single  planar  scene.  For more details, please refer to our RA-L [paper]().

## Dependencies
c++11, opencv, eigen, kindr, boost, glog, gflags

## Getting started
### Compilation
Please run the following commands to compile the code.
```bash
mkdir build
cd build
cmake ..
make
```
### Run
One example sequence can be downloaded [here](https://portland-my.sharepoint.com/:u:/g/personal/shanzhong4-c_ad_cityu_edu_hk/ESGCGKcTQflCm3JJ9jSYeG0B50ufNBB3Mzw6CBnRJ3yfrQ?e=Y74edp). Extract the files to the root directory of the project 
```
unzip example_seq.zip
``` 
and run:
```
./build/main
```


## Contact

If you have any questions, please contact me at 291790832@qq.com.
