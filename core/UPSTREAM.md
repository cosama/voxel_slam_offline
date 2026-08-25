# Upstream provenance and patch policy

## Pin

| | |
|---|---|
| Source | `upstream/Voxel-SLAM` (submodule, `git@github.com:hku-mars/Voxel-SLAM.git`) |
| Revision | `70fc8a28d63823d5989ff184daeea0787b672398` (`main`) |
| Package used | `VoxelSLAM/src/` |

The submodule is never modified. `git -C upstream/Voxel-SLAM status --short`
must stay empty. At configure time CMake copies `VoxelSLAM/src/` into
`build/upstream_staged` and applies the sorted patch series there. The local
algorithm delta is therefore exactly `patches/integration/`.

Patches are applied with `patch -F0 -p0`. A context mismatch is a configure
error rather than a fuzzy relocation, making upstream drift explicit. Patch
files are also registered with `CMAKE_CONFIGURE_DEPENDS`, so editing one
automatically reruns configuration and recreates the staged tree.

Because patches are grouped by upstream file, every hunk carries its rationale
at the integration site, tagged `voxelslam_bridge:`. After configuration,
`grep -rn 'voxelslam_bridge:' build/upstream_staged/` enumerates the complete
local integration surface in the source that is actually compiled. Each patch
header provides a numbered concern-level overview; vague or header-only
rationales are not accepted.

## Patch series

Patches are grouped one per upstream file so the complete delta for a file is
easy to audit:

| Patch | Upstream file | Purpose |
|---|---|---|
| `01-voxelslam-cpp.patch` | `src/voxelslam.cpp` | Captures odometry/map/loop/GBA results; hardens worker flags and ticket lifecycle; fixes gravity alignment, covariance rotation, stale slide-map pointers, and numerical guards; disables ROS subscriptions in Python; gates both output-directory sites on `is_save_map`. |
| `02-voxelslam-hpp.patch` | `src/voxelslam.hpp` | Carries LiDAR tickets through locked queues, exposes IMU wait state, preserves validated sweep timing, converts host-process exits to exceptions, and fixes covariance-frame rotation. |
| `03-voxel-map.patch` | `src/voxel_map.hpp` | Handles sparse residual workloads, guards LM ratios, converts invalid optimizer state to an exception, and fixes temporary-map sentinel comparisons. |
| `04-loop-refine.patch` | `src/loop_refine.hpp` | Rejects empty ICP inputs and records the exact eigenvalue/convergence verdict. |
| `05-ekf-imu.patch` | `src/ekf_imu.hpp` | Converts time regression to an exception and uses direction-normalized, fixed-magnitude gravity initialization. |
| `06-btc.patch` | `src/BTC.cpp` | Makes descriptor generation and geometric verification defined for empty or underspecified inputs. |
| `07-feature-point.patch` | `src/feature_point.hpp` | Converts unsupported LiDAR type termination into a catchable configuration error. |

Unchanged upstream files such as `BTC.h`, `preintegration.hpp`, and `tools.hpp`
come directly from the staged submodule copy and are not duplicated locally.

The public `VoxelSlamOptions` map-output controls (`save_path`, `bagname`, and
`is_save_map`) remain available. Both upstream directory-creation sites and
the bridge's parent-directory creation are conditional on `is_save_map`, so
the default offline API performs no embedded map-output filesystem writes.

## Upgrade procedure

1. Fetch upstream and select the new revision in `upstream/Voxel-SLAM`.
2. Reconfigure the core build. Any failed patch is an upstream API or behavior
   change that must be reviewed, not bypassed with fuzz.
3. Rebase each affected local file delta onto the new pristine source and
   regenerate its complete patch with `diff -U3` and `src/<file>` labels.
4. Build the Python extension and run the lifecycle/configuration and runner
   checks documented in `docs/TESTING.md`.
5. Confirm the upstream submodule remains clean, then update the revision in
   this file.
