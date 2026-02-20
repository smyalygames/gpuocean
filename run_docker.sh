#!/usr/bin/env sh

if [ -t 1 ]; then
    INTERACTIVE="-it"
else
    INTERACTIVE=""
fi

docker run \
    --rm \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    --privileged=true \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v .:/app \
    -v /app/.venv \
    -e ROCM_HOME=/opt/rocm \
    $(docker build -q .) \
    "$@"
