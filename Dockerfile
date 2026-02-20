FROM rocm/dev-ubuntu-24.04:7.0.2-complete AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# env variables for building
ENV MPICH_VERSION=3.4.3
ENV MPICH_URL="http://www.mpich.org/static/downloads/$MPICH_VERSION/mpich-$MPICH_VERSION.tar.gz"
ENV HDF5_VERSION=1.14.5
ENV HDF5_URL="https://github.com/HDFGroup/hdf5/releases/download/hdf5_$HDF5_VERSION/hdf5-$HDF5_VERSION.tar.gz"
ENV H5DIR=/usr
ENV NETCDF_VERSION=4.9.3
ENV NETCDF_URL="https://github.com/Unidata/netcdf-c/archive/refs/tags/v$NETCDF_VERSION.tar.gz"
ENV NCDIR=/usr
ENV VIRTUAL_ENV=/opt/.venv

# Update base os
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    file g++ gcc gfortran make gdb strace wget curl ca-certificates build-essential \
    zlib1g zlib1g-dev \
    m4 libxml2-dev libcurl4-openssl-dev zlib1g zlib1g-dev bzip2

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

RUN CC=mpicc ./configure --prefix=/usr --with-zlib=/usr --enable-parallel
RUN make -j$(nproc)
RUN make install
RUN ldconfig
WORKDIR /
RUN rm /hdf5-$HDF5_VERSION.tar.gz
RUN rm -rf /hdf5-$HDF5_VERSION

# Install netcdf
ADD $NETCDF_URL /netcdf-c-$NETCDF_VERSION.tar.gz
RUN tar xf /netcdf-c-$NETCDF_VERSION.tar.gz
WORKDIR /netcdf-c-$NETCDF_VERSION
RUN CC=mpicc CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=$NCDIR
RUN make -j$(nproc)
RUN make install


# Install dependencies
WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN git config --global --add safe.directory /app

# Cleanup
RUN apt-get clean
RUN rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

