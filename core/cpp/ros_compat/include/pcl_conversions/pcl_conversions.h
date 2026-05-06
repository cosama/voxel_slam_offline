#pragma once

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename MsgT>
void toROSMsg(const pcl::PointCloud<PointT>&, MsgT&) {}

template <typename MsgT, typename PointT>
void fromROSMsg(const MsgT&, pcl::PointCloud<PointT>& cloud) {
  cloud.clear();
}

}  // namespace pcl

