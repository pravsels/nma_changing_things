
xhost +local:
docker run --rm -it --gpus all --network=host --privileged \
    -v "$(pwd)":/app \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    nct:latest bash

