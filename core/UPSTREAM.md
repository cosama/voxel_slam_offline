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

Patches are grouped by upstream file. Headers state the integration need;
source comments are kept only when the code cannot express the constraint.

## Patch series

Patches are grouped one per upstream file so the complete delta for a file is
easy to audit:

| Patch | Upstream file | Purpose |
|---|---|---|
| `01-voxelslam-cpp.patch` | `src/voxelslam.cpp` | Captures results, synchronizes workers, and guards shared state. |
| `02-voxelslam-hpp.patch` | `src/voxelslam.hpp` | Carries LiDAR tickets through locked queues, exposes IMU wait state, preserves validated sweep timing, converts host-process exits to exceptions, and fixes covariance-frame rotation. |
| `03-voxel-map.patch` | `src/voxel_map.hpp` | Adds prior factors, handles sparse workloads, guards LM ratios, and fixes invalid state and sentinel handling. |
| `04-loop-refine.patch` | `src/loop_refine.hpp` | Rejects empty ICP inputs and records the exact eigenvalue/convergence verdict. |
| `05-ekf-imu.patch` | `src/ekf_imu.hpp` | Adds prior deskew, converts time regression to an exception, and normalizes gravity initialization. |
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
