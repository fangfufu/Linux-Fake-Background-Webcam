import signal
import sys
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict

import cv2
import numpy as np
import pyfakewebcam
import requests


class FakeCam:
    def __init__(
        self,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        scale_factor: float = 0.5,
        process_foreground: bool = False,
        bodypix_url: str = "http://127.0.0.1:9000",
        background_image: str = "background.jpg",
        foreground_image: str = "foreground.jpg",
        foreground_mask_image: str = "foreground-mask.png",
    ) -> None:
        self.process_foreground = process_foreground
        self.background_image = background_image
        self.foreground_image = foreground_image
        self.foreground_mask_image = foreground_mask_image
        self.fps = fps
        self.height = height
        self.width = width
        self.scale_factor = scale_factor
        self.bodypix_url = bodypix_url
        self.real_cam = cv2.VideoCapture("/dev/video0")
        self._setup_real_cam_properties()
        self.fake_cam = pyfakewebcam.FakeWebcam("/dev/video2", self.width, self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.session = requests.Session()
        self.images: Dict[str, Any] = {}

    def _setup_real_cam_properties(self):
        self.real_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.real_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.real_cam.set(cv2.CAP_PROP_FPS, self.fps)
        # In case the real webcam does not support the requested mode.
        self.height = int(self.real_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.real_cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    def _get_mask(self, frame):
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        _, data = cv2.imencode(".png", frame)
        r = self.session.post(
            url=self.bodypix_url, data=data.tobytes(), headers={"Content-Type": "application/octet-stream"}
        )
        mask = np.frombuffer(r.content, dtype=np.uint8)
        mask = mask.reshape((frame.shape[0], frame.shape[1]))
        mask = cv2.resize(
            mask, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor, interpolation=cv2.INTER_NEAREST
        )
        mask = cv2.dilate(mask, np.ones((20, 20), np.uint8), iterations=1)
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

    def load_images(self):
        background = cv2.imread(self.background_image)
        self.images["background"] = cv2.resize(background, (self.width, self.height))
        if self.process_foreground:
            foreground = cv2.imread(self.foreground_image)
            self.images["foreground"] = cv2.resize(foreground, (self.width, self.height))
            foreground_mask = cv2.imread(self.foreground_mask_image)
            foreground_mask = cv2.normalize(
                foreground_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            foreground_mask = cv2.resize(foreground_mask, (self.width, self.height))
            self.images["foreground_mask"] = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
            self.images["inverted_foreground_mask"] = 1 - self.images["foreground_mask"]

    def get_frame(self):
        _, frame = self.real_cam.read()
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = self._get_mask(frame)
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")

        # composite the foreground and background
        for c in range(frame.shape[2]):
            frame[:, :, c] = frame[:, :, c] * mask + self.images["background"][:, :, c] * (1 - mask)

        if self.process_foreground:
            for c in range(frame.shape[2]):
                frame[:, :, c] = (
                    frame[:, :, c] * self.images["inverted_foreground_mask"]
                    + self.images["foreground"][:, :, c] * self.images["foreground_mask"]
                )

        return frame

    def fake_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def run(self):
        while True:
            frame = self.get_frame()
            self.fake_frame(frame)


def parse_args():
    parser = ArgumentParser(description="Fake cam")
    parser.add_argument(
        "-p", "--process-foreground", default=False, action="store_true", help="Enable foreground processing"
    )
    parser.add_argument("-f", "--fps", default=30, type=int, help="How many FPS to process")
    parser.add_argument("-w", "--width", default=1280, type=int, help="Camera width")
    parser.add_argument("-H", "--height", default=720, type=int, help="Camera height")
    parser.add_argument("-s", "--scale-factor", default=0.5, type=float, help="Scale factor")
    parser.add_argument("-b", "--bodypix-url", default="http://127.0.0.1:9000", help="Tensorflow BodyPix URL")
    parser.add_argument("-B", "--background-image", default="background.jpg", help="Background image path")
    parser.add_argument("-F", "--foreground-image", default="foreground.jpg", help="Foreground image path")
    parser.add_argument(
        "-M", "--foreground-mask-image", default="foreground-mask.png", help="Foreground mask image path"
    )
    return parser.parse_args()


def sigint_handler(cam, signal, frame):
    print("Killing fake cam process")
    sys.exit(0)


def main():
    args = parse_args()
    cam = FakeCam(
        fps=args.fps,
        width=args.width,
        height=args.height,
        scale_factor=args.scale_factor,
        process_foreground=args.process_foreground,
        bodypix_url=args.bodypix_url,
        background_image=args.background_image,
        foreground_image=args.foreground_image,
        foreground_mask_image=args.foreground_mask_image,
    )
    cam.load_images()
    signal.signal(signal.SIGINT, partial(sigint_handler, cam))
    print("Running...")
    print("Please CTRL-C to exit")
    # frames forever
    cam.run()


if __name__ == "__main__":
    main()
