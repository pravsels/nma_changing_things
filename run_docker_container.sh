
os=$(uname)
gpu_flag=""

if [ "$os" != "Darwin" ]; then
    # if not on Mac, include --gpus all
    gpu_flag="--gpus all"
fi

xhost +local:
docker run --rm -it ${gpu_flag} --network=host --privileged \
    -v "$(pwd)":/app \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    nct:latest bash

