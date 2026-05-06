ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libeigen3-dev \
        libgtsam-dev \
        libpcl-dev \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

RUN python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /opt/build-venv/bin/pip wheel --no-cache-dir --wheel-dir /wheelhouse ./core ./runner

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/voxelslam/bin:${PATH}
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgtsam4 \
        libpcl-io1.14 \
        libpcl-kdtree1.14 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheelhouse /wheelhouse

RUN python3 -m venv /opt/voxelslam \
    && /opt/voxelslam/bin/pip install --no-cache-dir --no-index --find-links=/wheelhouse voxel-slam-offline voxel-slam-runner \
    && rm -rf /wheelhouse

WORKDIR /data

CMD ["python", "-c", "import voxelslam; print('voxel-slam-offline ready')"]
