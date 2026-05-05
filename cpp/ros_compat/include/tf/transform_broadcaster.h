#pragma once

#include <string>

#include <ros/ros.h>

namespace tf {

class Vector3 {
 public:
  Vector3(double, double, double) {}
};

class Quaternion {
 public:
  void setW(double) {}
  void setX(double) {}
  void setY(double) {}
  void setZ(double) {}
};

class Transform {
 public:
  void setOrigin(const Vector3&) {}
  void setRotation(const Quaternion&) {}
};

class StampedTransform {
 public:
  StampedTransform(const Transform&, const ros::Time&, const std::string&, const std::string&) {}
};

class TransformBroadcaster {
 public:
  void sendTransform(const StampedTransform&) {}
};

}  // namespace tf

