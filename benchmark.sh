#!/bin/sh
set -eu

width=1920
height=1080
fps=60
timeout=180
video_url="https://google.github.io/mediapipe/images/selfie_segmentation_web.mp4"

for ffmpeg in ffmpeg avconv ""; do
    test -n "$ffmpeg" || {
        echo "This script needs ffmpeg or avconv."
        exit 1
    }
    command -v "$ffmpeg" >/dev/null && break || :
done

# get sudo password and keep it active
sudo -v
( while sleep 1; do sudo -v; done; ) &
pids="$!"

# create file for test video
video="$(mktemp)"

# registister cleanup function
cleanup() {
    rm -f -- "$video"
    kill $pids; wait
    sudo -- $SHELL -c 'modprobe -r v4l2loopback; modprobe v4l2loopback'
}
trap cleanup EXIT

# create the necessary v4l2loopback devices
sudo -- $SHELL -s <<EOF
modprobe -r v4l2loopback
modprobe v4l2loopback devices=2 exclusive_caps=1,1 video_nr=99,100
EOF

# download video and loop it to /dev/video99
wget -qO "$video" "$video_url"
"$ffmpeg" -hide_banner -loglevel quiet -hwaccel auto \
    -re -stream_loop -1 -i "$video" \
    -vf scale="$width:$height" -r "$fps" \
    -f v4l2 -an -vcodec rawvideo -pix_fmt yuyv422 /dev/video99 &
pids="$pids $!"
sleep 3

git_head="$(git rev-parse HEAD)"
{
echo "${width}x${height}@$fps for $timeout seconds"
test $# -eq 0 || echo "extra arguments: $*"
echo "HEAD: $git_head"; echo

# system information
uname -srvmo
sed -n 's/^model name\s*:\s*//p' /proc/cpuinfo | sort | uniq -c; echo

# python information
python3 -c '
import importlib.metadata, sys
venv = (hasattr(sys, "real_prefix") or
    (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
print("python", sys.version)
print("virtualenv", venv)
print()
for module in sys.argv[1:]:
    try:
        print(module, importlib.metadata.version(module))
    except importlib.metadata.PackageNotFoundError:
        print(module, "not found")
' $(sed -rn 's/^([A-Z0-9][A-Z0-9._-]*[A-Z0-9]).*/\1/pi' requirements.txt); echo

# run fake.py for specified timeout and gather statistics
export PYTHONUNBUFFERED=yes
timeout --foreground --preserve-status -s QUIT "$timeout" \
    ./fake.py -w /dev/video99 -v /dev/video100 \
        -W "$width" -H "$height" -F "$fps" --no-ondemand "$@" |
    stdbuf -oL tr '\r' '\n' |
    python3 -c '
import sys, statistics
fps = []
for line in sys.stdin:
    if line.startswith("FPS:"):
        fps.append(float(line.split(":")[-1].strip()))
        print(fps[-1], end="\r", file=sys.stderr)
print("avg %.2f, stdev %.2f" %(statistics.mean(fps), statistics.stdev(fps)))
'
} | tee "benchmark.$git_head.txt"
