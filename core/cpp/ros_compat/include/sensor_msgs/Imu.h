#pragma once

#include <memory>

#include <ros/ros.h>

namespace sensor_msgs {

struct Vector3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Imu {
  using Ptr = std::shared_ptr<Imu>;
  using ConstPtr = std::shared_ptr<const Imu>;

  ros::Header header;
  Vector3 angular_velocity;
  Vector3 linear_acceleration;
};

using ImuPtr = Imu::Ptr;
using ImuConstPtr = Imu::ConstPtr;

}  // namespace sensor_msgs

