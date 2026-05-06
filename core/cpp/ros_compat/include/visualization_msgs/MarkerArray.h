#pragma once

#include <vector>

#include <visualization_msgs/Marker.h>

namespace visualization_msgs {
struct MarkerArray {
  std::vector<Marker> markers;
};
}  // namespace visualization_msgs

