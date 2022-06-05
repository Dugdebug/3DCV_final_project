docker run -it --rm --runtime nvidia --security-opt seccomp=unconfined --network host -e DISPLAY=$DISPLAY --name 3dcv_final \
    -v ~/3dCV:/3dCV \
    -v ~/RealtimeStereo:/RealtimeStereo \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    481 /bin/bash
