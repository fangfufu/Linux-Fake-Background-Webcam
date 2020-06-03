from fcntl import ioctl
import pyfakewebcam.v4l2 as v4l2
import os
import threading
from queue import Queue
import cv2
import numpy as np

class AkvCameraWriter:
    def __init__(self, webcam, width, height):
        self.webcam = webcam
        self.width = width
        self.height = height
        self.d = self.open_camera()
        self.queue = Queue(maxsize=1)
        self.thread = threading.Thread(target=self.writer_thread)
        self.thread.start()

    def open_camera(self):
        d = os.open(self.webcam, os.O_RDWR)
        cap = v4l2.v4l2_capability()
        ioctl(d, v4l2.VIDIOC_QUERYCAP, cap)
        vid_format = v4l2.v4l2_format()
        vid_format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        vid_format.fmt.pix.width = self.width
        vid_format.fmt.pix.height = self.height
        vid_format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_RGB24
        vid_format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        vid_format.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_SRGB
        ioctl(d, v4l2.VIDIOC_S_FMT, vid_format)
        return d

    def writer_thread(self):
        while True:
            elem = self.queue.get()
            if elem is None:
                break
            image_data = cv2.resize(elem, (self.width, self.height)).tobytes()
            try:
                os.write(self.d, image_data)
            except Exception:
                break

    def schedule_frame(self, image):
        self.queue.put(image)

    def __del__(self):
        self.queue.put(None)
        os.close(self.d)


if __name__ == "__main__":
    camera_w, camera_h = 1280, 720
    writer = AkvCameraWriter("/dev/video3", camera_w, camera_h)
    image = cv2.imread("background.jpg")
    iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    while True:
        writer.schedule_frame(image)
