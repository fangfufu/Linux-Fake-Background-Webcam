#!/usr/bin/env python3

from inotify_simple import INotify, flags
import itertools
import signal
import sys
import argparse
from functools import partial
from typing import Any, Dict
import cv2
import numpy as np
import pyfakewebcam
import os
import fnmatch
import time
import mediapipe as mp
from cmapy import cmap


class RealCam:
    def __init__(self, src, frame_width, frame_height, frame_rate, codec):
        self.cam = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.get_camera_values("original")
        c1, c2, c3, c4 = get_codec_args_from_string(codec)
        self._set_codec(cv2.VideoWriter_fourcc(c1, c2, c3, c4))
        self._set_frame_dimensions(frame_width, frame_height)
        self._set_frame_rate(frame_rate)
        self.get_camera_values("new")

    def get_camera_values(self, status):
        print(
            "Real camera {} values are set as: {}x{} with {} FPS and video codec {}".format(
                status,
                self.get_frame_width(),
                self.get_frame_height(),
                self.get_frame_rate(),
                self.get_codec()
            )
        )

    def _set_codec(self, codec):
        self.cam.set(cv2.CAP_PROP_FOURCC, codec)
        if codec != self.get_codec():
            _log_camera_property_not_set(cv2.CAP_PROP_FOURCC, codec)

    def _set_frame_dimensions(self, width, height):
        # width/height need to both be set before checking for any errors.
        # If either are checked before setting both, either can be reported as
        # not set properly
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if width != self.get_frame_width():
            _log_camera_property_not_set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height != self.get_frame_height():
            _log_camera_property_not_set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def _set_frame_rate(self, fps):
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        if fps != self.get_frame_rate():
            _log_camera_property_not_set(cv2.CAP_PROP_FPS, fps)

    def get_codec(self):
        return int(self.cam.get(cv2.CAP_PROP_FOURCC))

    def get_frame_width(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_rate(self):
        return int(self.cam.get(cv2.CAP_PROP_FPS))

    def read(self):
        while True:
            grabbed, frame = self.cam.read()
            if not grabbed:
                continue
            return frame


class FakeCam:
    def __init__(self, args) -> None:
        self.no_background = args.no_background
        self.use_foreground = not args.no_foreground
        self.hologram = args.hologram
        self.tiling = args.tile_background
        self.background_blur = getNextOddNumber(args.background_blur)
        self.sigma = self.background_blur / args.background_blur_sigma_frac
        self.background_keep_aspect = args.background_keep_aspect
        self.image_folder = args.image_folder
        self.background_image = args.background_image
        self.foreground_image = args.foreground_image
        self.foreground_mask_image = args.foreground_mask_image
        self.webcam_path = args.webcam_path
        self.width = args.width
        self.height = args.height
        self.fps = args.fps
        self.codec = args.codec
        self.MRAR = getPercentageFloat(
            args.background_mask_update_speed) # Mask Running Average Ratio
        self.use_sigmoid = args.use_sigmoid
        self.threshold = getPercentageFloat(args.threshold)
        self.postprocess = args.no_postprocess
        self.ondemand = args.ondemand
        self.v4l2loopback_path = args.v4l2loopback_path
        self.classifier = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=args.select_model)

        # These do not involve reading from args
        self.old_mask = None
        self.real_cam = RealCam(self.webcam_path,
                                self.width,
                                self.height,
                                self.fps,
                                self.codec)
        # In case the real webcam does not support the requested mode.
        self.width = self.real_cam.get_frame_width()
        self.height = self.real_cam.get_frame_height()
        self.fake_cam = pyfakewebcam.FakeWebcam(self.v4l2loopback_path, self.width,
                                                self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.images: Dict[str, Any] = {}
        self.paused = False
        self.consumers = 0

    def resize_image(self, img, keep_aspect):
        """ Rescale image to dimensions self.width, self.height.

        If keep_aspect is True then scale & crop the image so that its pixels
        retain their aspect ratio.
        """
        if self.width == 0 or self.height == 0:
            raise RuntimeError("Camera dimensions error w={} h={}".format(
                self.width, self.height))
        if keep_aspect:
            imgheight, imgwidth, = img.shape[:2]
            scale = max(self.width / imgwidth, self.height / imgheight)
            newimgwidth, newimgheight = int(np.floor(self.width / scale)), int(
                np.floor(self.height / scale))
            ix0 = int(np.floor(0.5 * imgwidth - 0.5 * newimgwidth))
            iy0 = int(np.floor(0.5 * imgheight - 0.5 * newimgheight))
            img = cv2.resize(img[iy0:iy0 + newimgheight, ix0:ix0 + newimgwidth, :],
                             (self.width, self.height))
        else:
            img = cv2.resize(img, (self.width, self.height))
        return img

    def load_images(self):
        self.images: Dict[str, Any] = {}

        background = cv2.imread(
            findFile(self.background_image, self.image_folder))
        if background is not None:
            if not self.tiling:
                background = self.resize_image(background,
                                               self.background_keep_aspect)
            else:
                sizey, sizex = background.shape[0], background.shape[1]
                if sizex > self.width and sizey > self.height:
                    background = cv2.resize(
                        background, (self.width, self.height))
                else:
                    repx = (self.width - 1) // sizex + 1
                    repy = (self.height - 1) // sizey + 1
                    background = np.tile(background, (repy, repx, 1))
                    background = background[0:self.height, 0:self.width]
            background = itertools.repeat(background)
        else:
            background_video = cv2.VideoCapture(
                findFile(self.background_image, self.image_folder))
            if not background_video.isOpened():
                raise RuntimeError("Couldn't open video '{}'".format(
                    self.background_image))
            self.bg_video_fps = background_video.get(cv2.CAP_PROP_FPS)
            # Initiate current fps to background video fps
            self.current_fps = self.bg_video_fps

            def read_frame():
                ret, frame = background_video.read()
                if not ret:
                    background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = background_video.read()
                    assert ret, 'cannot read frame %r' % self.background_image
                return self.resize_image(frame, self.background_keep_aspect)

            def next_frame():
                while True:
                    # Number of frames we need to advance background movie,
                    # fractional.
                    advrate = self.bg_video_fps / self.current_fps
                    if advrate < 1:
                        # Number of frames<1 so to avoid movie freezing randomly
                        # choose whether to advance by one frame with correct
                        # probability.
                        self.bg_video_adv_rate = 1 if np.random.uniform() < advrate else 0
                    else:
                        # Just round to nearest number of frames when >=1.
                        self.bg_video_adv_rate = round(advrate)
                    for i in range(self.bg_video_adv_rate):
                        frame = read_frame()
                    yield frame
            background = next_frame()

        self.images["background"] = background

        if self.use_foreground and self.foreground_image is not None:
            foreground = cv2.imread(
                findFile(self.foreground_image, self.image_folder))
            self.images["foreground"] = cv2.resize(foreground,
                                                   (self.width, self.height))
            foreground_mask = cv2.imread(
                findFile(self.foreground_mask_image, self.image_folder))
            foreground_mask = cv2.normalize(
                foreground_mask, None, alpha=0, beta=1,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            foreground_mask = cv2.resize(foreground_mask,
                                         (self.width, self.height))
            self.images["foreground_mask"] = cv2.cvtColor(
                foreground_mask, cv2.COLOR_BGR2GRAY)
            self.images["inverted_foreground_mask"] = 1 - \
                self.images["foreground_mask"]

    def compose_frame(self, frame):
        mask = self.classifier.process(frame).segmentation_mask

        if self.threshold < 1:
            mask = (mask > self.threshold) * mask

        if self.postprocess:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            mask = cv2.blur(mask.astype(float), (10, 10))

        if self.MRAR < 1:
            if self.old_mask is None:
                self.old_mask = mask
            mask = mask * self.MRAR + self.old_mask * (1.0 - self.MRAR)
            self.old_mask = mask

        # Get background image
        if self.no_background is False:
            background_frame = next(self.images["background"])
        else:
            background_frame = cv2.GaussianBlur(frame,
                                                (self.background_blur,
                                                 self.background_blur),
                                                self.sigma,
                                                borderType=cv2.BORDER_DEFAULT)

        # Add hologram to the person
        if self.hologram:
            frame = hologram_effect(frame)

        # Replace background
        if self.use_sigmoid:
            mask = sigmoid(mask)

        for c in range(frame.shape[2]):
            frame[:, :, c] = frame[:, :, c] * mask + \
                background_frame[:, :, c] * (1 - mask)

        # Add foreground if needed
        if self.use_foreground and self.foreground_image is not None:
            for c in range(frame.shape[2]):
                frame[:, :, c] = (
                    frame[:, :, c] * self.images["inverted_foreground_mask"] +
                    self.images["foreground"][:, :, c] *
                    self.images["foreground_mask"]
                )

        return frame

    def put_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def run(self):
        self.load_images()
        t0 = time.monotonic()
        print_fps_period = 1
        frame_count = 0
        blank_image = None

        inotify = INotify(nonblocking=True)
        if self.ondemand:
            watch_flags = flags.CREATE | flags.OPEN | flags.CLOSE_NOWRITE | flags.CLOSE_WRITE
            wd = inotify.add_watch(self.v4l2loopback_path, watch_flags)
            self.paused = True

        while True:
            if self.ondemand:
                for event in inotify.read(0):
                    for flag in flags.from_mask(event.mask):
                        if flag == flags.CLOSE_NOWRITE or flag == flags.CLOSE_WRITE:
                            self.consumers -= 1
                        if flag == flags.OPEN:
                            self.consumers += 1
                    if self.consumers > 0:
                        self.paused = False
                        self.load_images()
                        print("Consumers:", self.consumers)
                    else:
                        self.consumers = 0
                        self.paused = True
                        print("No consumers remaining, paused")

            if not self.paused:
                if self.real_cam is None:
                    self.real_cam = RealCam(self.webcam_path,
                                            self.width,
                                            self.height,
                                            self.fps,
                                            self.codec)
                frame = self.real_cam.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.compose_frame(frame)
                self.put_frame(frame)
                frame_count += 1
                td = time.monotonic() - t0
                if td > print_fps_period:
                    self.current_fps = frame_count / td
                    print("FPS: {:6.2f}".format(self.current_fps), end="\r")
                    frame_count = 0
                    t0 = time.monotonic()
            else:
                width = 0
                height = 0
                if self.real_cam is not None:
                    frame = self.real_cam.read()
                    self.real_cam = None
                    if blank_image is not None:
                        blank_image.flags.writeable = True
                    blank_image = np.zeros(frame.shape, dtype=np.uint8)
                    blank_image.flags.writeable = False
                self.put_frame(blank_image)
                time.sleep(1)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print("\nPaused.")
        else:
            print("\nResuming, reloading background / foreground images...")
            self.load_images()


def parse_args():
    parser = argparse.ArgumentParser(description="Faking your webcam background under \
                            GNU/Linux. Please refer to: \
                            https://github.com/fangfufu/Linux-Fake-Background-Webcam",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-W", "--width", default=1280, type=int,
                        help="Set real webcam width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Set real webcam height")
    parser.add_argument("-F", "--fps", default=30, type=int,
                        help="Set real webcam FPS")
    parser.add_argument("-C", "--codec", default='MJPG', type=str,
                        help="Set real webcam codec")
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-v", "--v4l2loopback-path", default="/dev/video2",
                        help="V4l2loopback device path")
    parser.add_argument("-i", "--image-folder", default=".",
                        help="Folder which contains foreground and background images")
    parser.add_argument("--no-background", action="store_true",
                        help="Disable background image and blur the real background")
    parser.add_argument("-b", "--background-image", default="background.*",
                        help="Background image path, animated background is \
                        supported.")
    parser.add_argument("--tile-background", action="store_true",
                        help="Tile the background image")
    parser.add_argument("--background-blur", default="21", type=int, metavar='k',
                        help="The gaussian bluring kernel size in pixels")
    parser.add_argument("--background-blur-sigma-frac", default="3", type=int, metavar='frac',
                        help="The fraction of the kernel size to use for the sigma value (ie. sigma = k / frac)")
    parser.add_argument("--background-keep-aspect", action="store_true",
                        help="Crop background if needed to maintain aspect ratio")
    parser.add_argument("--no-foreground", action="store_true",
                        help="Disable foreground image")
    parser.add_argument("-f", "--foreground-image", default="foreground.*",
                        help="Foreground image path")
    parser.add_argument("-m", "--foreground-mask-image",
                        default="foreground-mask.*",
                        help="Foreground mask image path")
    parser.add_argument("--hologram", action="store_true",
                        help="Add a hologram effect")
    parser.add_argument("--ondemand", action="store_true",
                        help="Pause processing when there is no application using the virtual webcam")
    parser.add_argument("--background-mask-update-speed", default="50", type=int,
                        help="The running average percentage for background mask updates")
    parser.add_argument("--use-sigmoid", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--threshold", default="75", type=int,
                        help="The minimum percentage threshold for accepting a pixel as foreground")
    parser.add_argument("--no-postprocess", action="store_false",
                        help="Disable postprocessing (masking dilation and blurring)")
    parser.add_argument("--select-model", default="1", type=int,
                        help="Select the model for MediaPipe. For more information, please refer to \
https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294")
    parser.add_argument("--cmapy-bg", default=None, type=str,
                        help="Apply colour map to background using cmapy")
    parser.add_argument("--cmapy-person", default=None, type=str,
                        help="Apply colour map to the person using cmapy")
    return parser.parse_args()


def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 3, 4
    for y in range(holo.shape[0]):
        if y % (bandLength + bandGap) < bandLength:
            holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(
        holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(
        holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out


def sigint_handler(cam, signal, frame):
    cam.toggle_pause()


def sigquit_handler(cam, signal, frame):
    print("\nKilling fake cam process")
    sys.exit(0)


def getNextOddNumber(number):
    if number % 2 == 0:
        return number + 1
    return number


def getPercentageFloat(number):
    return min(max(number, 0), 100) / 100.


def sigmoid(x, a=5., b=-10.):
    """
    Converts the 0-1 value to a sigmoid going from zero to 1 in the same range
    """
    z = np.exp(a + b * x)
    sig = 1 / (1 + z)
    return sig


def findFile(pattern, path):
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


def get_codec_args_from_string(codec):
    return (char for char in codec)


def _log_camera_property_not_set(prop, value):
    print("Cannot set camera property {} to {}. "
          "Defaulting to auto-detected property set by opencv".format(prop,
                                                                      value))


def main():
    args = parse_args()
    cam = FakeCam(args)
    signal.signal(signal.SIGINT, partial(sigint_handler, cam))
    signal.signal(signal.SIGQUIT, partial(sigquit_handler, cam))
    print("Running...")
    print("Please CTRL-C to pause and reload the background / foreground images")
    print("Please CTRL-\ to exit")
    # frames forever
    cam.run()


if __name__ == "__main__":
    main()
