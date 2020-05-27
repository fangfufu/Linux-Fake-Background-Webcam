import asyncio
import itertools
import signal
import sys
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict

import aiohttp
import cv2
import numpy as np
import pyfakewebcam
import requests
import os
import fnmatch
import time

def findFile(pattern, path):
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None

class FakeCam:
    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        scale_factor: float,
        use_foreground: bool,
        hologram: bool,
        bodypix_url: str,
        background_image: str,
        foreground_image: str,
        foreground_mask_image: str,
        webcam_path: str,
        v4l2loopback_path: str
    ) -> None:
        self.use_foreground = use_foreground
        self.hologram = hologram
        self.background_image = background_image
        self.foreground_image = foreground_image
        self.foreground_mask_image = foreground_mask_image
        self.fps = fps
        self.height = height
        self.width = width
        self.scale_factor = scale_factor
        self.bodypix_url = bodypix_url
        self.real_cam = cv2.VideoCapture(webcam_path,cv2.CAP_V4L2)
        self._setup_real_cam_properties()
        self.fake_cam = pyfakewebcam.FakeWebcam(v4l2loopback_path, self.width,
                                                self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.session = requests.Session()
        self.images: Dict[str, Any] = {}
        self.lock = asyncio.Lock()

    def _setup_real_cam_properties(self):
        self.real_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.real_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.real_cam.set(cv2.CAP_PROP_FPS, self.fps)
        # In case the real webcam does not support the requested mode.
        self.height = int(self.real_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.real_cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    async def _get_mask(self, frame, session):
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor,
                           fy=self.scale_factor)
        _, data = cv2.imencode(".png", frame)
        async with session.post(
            url=self.bodypix_url, data=data.tostring(),
            headers={"Content-Type": "application/octet-stream"}
        ) as r:
            mask = np.frombuffer(await r.read(), dtype=np.uint8)
            mask = mask.reshape((frame.shape[0], frame.shape[1]))
            mask = cv2.resize(
                mask, (0, 0), fx=1 / self.scale_factor,
                fy=1 / self.scale_factor, interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
            mask = cv2.blur(mask.astype(float), (30, 30))
            return mask

    def shift_image(self, img, dx, dy):
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

    async def load_images(self):
        async with self.lock:
            self.images: Dict[str, Any] = {}

            background = cv2.imread(self.background_image)
            if background is not None:
                background = cv2.resize(background, (self.width, self.height))
                background = itertools.repeat(background)
            else:
                background_video = cv2.VideoCapture(self.background_image)
                def iter_frames():
                    while True:
                        ret, frame = background_video.read()
                        if not ret:
                            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = background_video.read()
                            assert ret, 'cannot read frame %r' % self.background_image
                        frame = cv2.resize(frame, (self.width, self.height))
                        yield frame
                background = iter_frames()
            self.images["background"] = background

            if self.use_foreground and self.foreground_image is not None:
                foreground = cv2.imread(self.foreground_image)
                self.images["foreground"] = cv2.resize(foreground,
                                                       (self.width, self.height))
                foreground_mask = cv2.imread(self.foreground_mask_image)
                foreground_mask = cv2.normalize(
                    foreground_mask, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                foreground_mask = cv2.resize(foreground_mask,
                                             (self.width, self.height))
                self.images["foreground_mask"] = cv2.cvtColor(
                    foreground_mask, cv2.COLOR_BGR2GRAY)
                self.images["inverted_foreground_mask"] = 1 - self.images["foreground_mask"]

    def hologram_effect(self, img):
        # add a blue tint
        holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        # add a halftone effect
        bandLength, bandGap = 2, 3
        for y in range(holo.shape[0]):
            if y % (bandLength+bandGap) < bandLength:
                holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, self.shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4, self.shift_image(holo.copy(), -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
        return out 

    
    async def get_frame(self, session):
        _, frame = self.real_cam.read()
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = await self._get_mask(frame, session)
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")
                traceback.print_exc()     
      
        async with self.lock:
            if self.hologram: 
                frame = self.hologram_effect(frame)

            # composite the foreground and background
            background = next(self.images["background"])
            for c in range(frame.shape[2]):
                frame[:, :, c] = frame[:, :, c] * mask + background[:, :, c] * (1 - mask)

        if self.use_foreground and self.foreground_image is not None:
            async with self.lock:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = (
                        frame[:, :, c] * self.images["inverted_foreground_mask"]
                        + self.images["foreground"][:, :, c] * self.images["foreground_mask"]
                    )

        return frame

    def fake_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    async def run(self):
        await self.load_images()
        async with aiohttp.ClientSession() as session:
            t0 = time.monotonic()
            print_fps_period = 1
            frame_count = 0
            while True:
                frame = await self.get_frame(session)
                self.fake_frame(frame)
                frame_count += 1
                td = time.monotonic() - t0
                if td > print_fps_period:
                    print("FPS: {:6.2f}".format(frame_count / td), end="\r")
                    frame_count = 0
                    t0 = time.monotonic()

def parse_args():
    parser = ArgumentParser(description="Faking your webcam background under \
                            GNU/Linux. Please make sure your bodypix network \
                            is running. For more information, please refer to: \
                            https://github.com/fangfufu/Linux-Fake-Background-Webcam")
    parser.add_argument("-p", "--no-foreground",
                        default=False, action="store_true",
                        help="Disable foreground image")
    parser.add_argument("-w", "--width", default=1280, type=int,
                        help="Camera width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Camera height")
    parser.add_argument("-s", "--scale-factor", default=0.5, type=float,
                        help="Scale factor")
    parser.add_argument("-b", "--bodypix-url", default="http://127.0.0.1:9000",
                        help="Tensorflow BodyPix URL")
    parser.add_argument("-I", "--image-folder", default=".",
                        help="Background and foreground images folder.")
    parser.add_argument("-B", "--background-image", default="background.*",
                        help="Background image path, animated background is \
                        supported.")
    parser.add_argument("-F", "--foreground-image", default="foreground.*",
                        help="Foreground image path")
    parser.add_argument("-M", "--foreground-mask-image",
                        default="foreground-mask.*",
                        help="Foreground mask image path")
    parser.add_argument("-W", "--webcam-path", default="/dev/video0",
                        help="Webcam path")
    parser.add_argument("-V", "--v4l2loopback-path", default="/dev/video2",
                        help="V4l2loopback device path")
    parser.add_argument("--hologram", default=False, action="store_true", help="Add a hologram effect") 
    return parser.parse_args()


def sigint_handler(loop, cam, signal, frame):
    print("Reloading background / foreground images")
    asyncio.ensure_future(cam.load_images())


def sigquit_handler(loop, cam, signal, frame):
    print("Killing fake cam process")
    sys.exit(0)


def main():
    args = parse_args()
    cam = FakeCam(
        fps=args.fps,
        width=args.width,
        height=args.height,
        scale_factor=args.scale_factor,
        use_foreground=not args.no_foreground,
        hologram=args.hologram,
        bodypix_url=args.bodypix_url,
        background_image=findFile(args.background_image, args.image_folder),
        foreground_image=findFile(args.foreground_image, args.image_folder),
        foreground_mask_image=findFile(args.foreground_mask_image, args.image_folder),
        webcam_path=args.webcam_path,
        v4l2loopback_path=args.v4l2loopback_path)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sigint_handler, loop, cam))
    signal.signal(signal.SIGQUIT, partial(sigquit_handler, loop, cam))
    print("Running...")
    print("Please CTRL-C to reload the background / foreground images")
    print("Please CTRL-\ to exit")
    # frames forever
    loop.run_until_complete(cam.run())


if __name__ == "__main__":
    main()
