docker run -it --privileged \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e LOCAL_USER_ID="$(id -u)" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev:/dev:rw \
    -v /home/a/clothoid_seminar/seminar:/home/user/seminar \
    --hostname $(hostname) \
    --network host \
    --name clothoid_seminar \
    rth0824/autonomous-racing-simulator:ver1.1 bash
