#pragma once

#include <any>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifndef ROS_WARN
#define ROS_WARN(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

namespace ros {

class Time {
 public:
  Time() = default;
  explicit Time(double seconds) : seconds_(seconds) {}

  double toSec() const { return seconds_; }
  void fromSec(double seconds) { seconds_ = seconds; }

  static Time now() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return Time(std::chrono::duration<double>(now).count());
  }

 private:
  double seconds_ = 0.0;
};

struct Header {
  Time stamp;
  std::string frame_id;
  uint32_t seq = 0;
};

class Publisher {
 public:
  template <typename T>
  void publish(const T&) const {}
};

class Subscriber {};

class NodeHandle {
 public:
  NodeHandle() = default;

  bool ok() const {
    bool shutdown = false;
    param("__shutdown", shutdown, false);
    return !shutdown;
  }

  template <typename T>
  void setParam(const std::string& key, const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    params_[key] = value;
  }

  template <typename T>
  void param(const std::string& key, T& value, const T& default_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = params_.find(key);
    if (it == params_.end()) {
      value = default_value;
      return;
    }
    if (const T* typed = std::any_cast<T>(&it->second)) {
      value = *typed;
      return;
    }
    value = default_value;
  }

  template <typename MsgT, typename CallbackT>
  Subscriber subscribe(const std::string&, int, CallbackT) {
    return Subscriber();
  }

  template <typename MsgT>
  Publisher advertise(const std::string&, int) {
    return Publisher();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::any> params_;
};

inline void init(int, char**, const std::string&) {}
inline void spin() {}
inline void spinOnce() {}

}  // namespace ros
