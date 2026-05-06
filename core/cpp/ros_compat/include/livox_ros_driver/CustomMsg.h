#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <ros/ros.h>

namespace livox_ros_driver {

struct CustomPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t reflectivity = 0;
  uint32_t offset_time = 0;
};

struct CustomMsg {
  using Ptr = std::shared_ptr<CustomMsg>;
  using ConstPtr = std::shared_ptr<const CustomMsg>;

  ros::Header header;
  uint32_t point_num = 0;
  std::vector<CustomPoint> points;
};

}  // namespace livox_ros_driver

