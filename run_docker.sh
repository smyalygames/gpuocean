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
    --user=1000 \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v .:/app \
    -v /app/.venv \
    -e ROCM_HOME=/opt/rocm \
    -p 12345:12345 \
    $(docker build -q .) \
    "$@"
