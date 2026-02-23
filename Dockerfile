FROM rocm/dev-ubuntu-24.04:7.0.2-complete AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# env variables for building
ENV MPICH_VERSION=3.4.3
ENV MPICH_URL="http://www.mpich.org/static/downloads/$MPICH_VERSION/mpich-$MPICH_VERSION.tar.gz"
ENV HDF5_VERSION=2.0.0
ENV HDF5_URL="https://github.com/HDFGroup/hdf5/releases/download/$HDF5_VERSION/hdf5-$HDF5_VERSION.tar.gz"
ENV H5DIR=/usr
ENV NETCDF_VERSION=4.9.3
ENV NETCDF_URL="https://github.com/Unidata/netcdf-c/archive/refs/tags/v$NETCDF_VERSION.tar.gz"
ENV NCDIR=/usr
ENV VIRTUAL_ENV=/opt/.venv

# Update base os
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    file g++ gcc gfortran make gdb strace wget curl git ca-certificates build-essential cmake \
    zlib1g zlib1g-dev \
    m4 libxml2 libxml2-utils libxml2-dev libcurl4-openssl-dev zlib1g zlib1g-dev bzip2 libbz2-dev libbz2-1.0 libzip-dev

# Download and build MPICH
ADD $MPICH_URL /mpich-$MPICH_VERSION.tar.gz
RUN tar xf /mpich-$MPICH_VERSION.tar.gz
WORKDIR /mpich-$MPICH_VERSION

RUN ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr --with-device=ch3 --with-hip=/opt/rocm
RUN make -j$(nproc)
RUN make install
RUN ldconfig
WORKDIR /
RUN rm /mpich-$MPICH_VERSION.tar.gz
RUN rm -rf /mpich-$MPICH_VERSION

# Install HDF5
ADD $HDF5_URL /hdf5-$HDF5_VERSION.tar.gz
RUN tar xf /hdf5-$HDF5_VERSION.tar.gz
WORKDIR /hdf5-$HDF5_VERSION

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=None \
    -DCMAKE_INSTALL_PREFIX=${H5DIR} \
    -Wno-dev \
    -DHDF5_USE_GNU_DIRS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DHDF5_BUILD_CPP_LIB=ON \
    -DHDF5_BUILD_HL_LIB=ON \
    -DHDF5_BUILD_FORTRAN=OFF \
    -DHDF5_BUILD_JAVA=OFF \
    -DHDF5_ENABLE_ZLIB_SUPPORT=ON \
    -DHDF5_INSTALL_CMAKE_DIR=lib/cmake/hdf5 \
    -DALLOW_UNSUPPORTED=ON \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_ALLOW_UNSUPPORTED=ON

RUN cmake --build build
RUN cmake --install build
RUN ldconfig
WORKDIR /
RUN rm /hdf5-$HDF5_VERSION.tar.gz
RUN #rm -rf /hdf5-$HDF5_VERSION

# Install netcdf
# TODO revert once netCDF 4.10.0 is released.
RUN git clone --single-branch --branch fix-hdf5-2.0.0.wif https://github.com/Unidata/netcdf-c.git /netcdf-c
WORKDIR /netcdf-c
RUN cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${NCDIR} \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DNETCDF_WITH_PLUGIN_DIR=${NCDIR}/lib/netcdf/plugin \
    -DNETCDF_ENABLE_HDF5=ON \
    -DHDF5_PARALLEL=${H5DIR} \
    -DNETCDF_ENABLE_PARALLEL4=ON \
    -DNETCDF_ENABLE_CDF5=ON \
    -DENABLE_DAP_LONG_TESTS=ON \
    -DENABLE_DAP_REMOTE_TESTS=ON \
    -DENABLE_EXAMPLE_TESTS=ON \
    -DENABLE_EXTRA_TESTS=ON \
    -DNETCDF_ENABLE_FILTER_TESTING=ON \
    -DNETCDF_ENABLE_LARGE_FILE_TESTS=ON \
    -DNETCDF_ENABLE_UNIT_TESTS=ON \
    -DNETCDF_ENABLE_LOGGING=ON \
    -DENABLE_PLUGIN_INSTALL=ON \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc
RUN cmake --build build
RUN cmake --install build
WORKDIR /
RUN rm -rf /netcdf-c


# Install dependencies
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN git config --global --add safe.directory /app

# Cleanup
RUN apt-get clean
RUN rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

