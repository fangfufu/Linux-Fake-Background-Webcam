#!/usr/bin/env python3

from inotify_simple import INotify, flags
import itertools
import signal
import os
import sys
import configargparse
from functools import partial
from typing import Any, Dict
import cv2
import numpy as np
import pyfakewebcam
import time
from cmapy import cmap
from matplotlib import colormaps
import copy

class RealCam:
    def __init__(self, src, frame_width, frame_height, frame_rate, codec):
        self.cam = cv2.VideoCapture(src, cv2.CAP_V4L2)
        while not self.cam.isOpened():
            print("Failed to open camera: {}. retrying...".format(src))
            time.sleep(1)
            self.cam.open(src, cv2.CAP_V4L2)
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
        self.webcam_path = args.webcam_path
        self.width = args.width
        self.height = args.height
        self.fps = args.fps
        self.codec = args.codec
        self.use_sigmoid = args.use_sigmoid
        self.threshold = getPercentageFloat(args.threshold)
        self.postprocess = args.no_postprocess
        self.ondemand = not args.no_ondemand
        self.v4l2loopback_path = args.v4l2loopback_path

        # Process unified filter arguments with structured defaults
        self.filters = {
            'selfie': create_filter_config(process_filter_args(args.selfie), 'selfie'),
            'background': create_filter_config(process_filter_args(args.background), 'background'),
            'mask': create_filter_config(process_filter_args(args.mask), 'mask')
        }

        # These do not involve reading from args
        self.old_mask = None
        self.real_cam = RealCam(self.webcam_path,
                                self.width,
                                self.height,
                                self.fps,
                                self.codec)
        # In case the real webcam does not support the requested mode.
        self.real_width = self.real_cam.get_frame_width()
        self.real_height = self.real_cam.get_frame_height()
        self.fake_cam = pyfakewebcam.FakeWebcam(self.v4l2loopback_path, self.width,
                                                self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.images: Dict[str, Any] = {}
        self.paused = False
        self.consumers = 0

        if args.dump:
            print("internal state for debugging:")
            print(self.__dict__)
            sys.exit(0)

        # slow model loading
        import mediapipe as mp
        self.classifier = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=args.select_model)


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

        # Process background filters
        bg_config = self.filters['background']
        background = None

        # Check for 'no' background
        if bg_config['disabled']:
            pass
        elif bg_config['file']:
            bg_path = bg_config['file']
            background = cv2.imread(bg_path)
            if background is not None:
                # Static image
                if not bg_config['tile']:
                    background = self.resize_image(background, bg_config['crop'])
                else:
                    sizey, sizex = background.shape[0], background.shape[1]
                    if sizex > self.width and sizey > self.height:
                        background = cv2.resize(background, (self.width, self.height))
                    else:
                        repx = (self.width - 1) // sizex + 1
                        repy = (self.height - 1) // sizey + 1
                        background = np.tile(background, (repy, repx, 1))
                        background = background[0:self.height, 0:self.width]
                background = itertools.repeat(background)
            else:
                # Try as video
                background_video = cv2.VideoCapture(bg_path)
                if not background_video.isOpened():
                    raise RuntimeError("Couldn't open file '{}'".format(bg_path))
                self.bg_video_fps = background_video.get(cv2.CAP_PROP_FPS)
                self.current_fps = self.bg_video_fps

                def read_frame():
                    ret, frame = background_video.read()
                    if not ret:
                        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = background_video.read()
                        assert ret, 'cannot read frame %r' % bg_path
                    return self.resize_image(frame, bg_config['crop'])

                def next_frame():
                    while True:
                        advrate = self.bg_video_fps / self.current_fps
                        if advrate < 1:
                            # Use deterministic value if TEST_DETERMINISTIC env var is set
                            if os.environ.get('TEST_DETERMINISTIC'):
                                self.bg_video_adv_rate = 1 if 0.5 < advrate else 0  # Fixed threshold
                            else:
                                self.bg_video_adv_rate = 1 if np.random.uniform() < advrate else 0
                        else:
                            self.bg_video_adv_rate = round(advrate)
                        for i in range(self.bg_video_adv_rate):
                            frame = read_frame()
                        yield frame
                background = next_frame()

        self.images["background"] = background

        # Process mask filters
        mask_config = self.filters['mask']

        # Check for 'no' mask (disabled)
        if not mask_config['disabled']:
            if mask_config['file']:
                # Check if it's a mask file or foreground image file
                mask_path = mask_config['file']

                # Try to determine if this is a mask or a foreground image
                # Masks typically have transparency or are grayscale
                test_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if test_img is not None:
                    # Check if it's likely a mask (grayscale or has alpha channel)
                    is_mask = len(test_img.shape) == 2 or (len(test_img.shape) == 3 and test_img.shape[2] == 4)

                    if is_mask or mask_config.get('mask_file'):
                        # It's a mask file
                        foreground_mask = cv2.imread(mask_path)
                        if foreground_mask is not None:
                            foreground_mask = cv2.normalize(
                                foreground_mask, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            foreground_mask = cv2.resize(foreground_mask,
                                                         (self.width, self.height))
                            self.images["foreground_mask"] = cv2.cvtColor(
                                foreground_mask, cv2.COLOR_BGR2GRAY)
                            self.images["inverted_foreground_mask"] = 1 - \
                                self.images["foreground_mask"]
                    else:
                        # It's a foreground image - create a default mask
                        foreground = cv2.imread(mask_path)
                        if foreground is not None:
                            self.images["foreground"] = cv2.resize(foreground,
                                                                   (self.width, self.height))
                            # Create a default mask (full opacity)
                            self.images["foreground_mask"] = np.ones((self.height, self.width),
                                                                     dtype=np.float32)
                            self.images["inverted_foreground_mask"] = np.zeros((self.height, self.width),
                                                                               dtype=np.float32)

            # Check for separate foreground image
            if mask_config.get('foreground_file'):
                foreground = cv2.imread(mask_config['foreground_file'])
                if foreground is not None:
                    self.images["foreground"] = cv2.resize(foreground,
                                                           (self.width, self.height))

    def compose_frame(self, frame):
        mask = copy.copy(self.classifier.process(frame).segmentation_mask)

        if self.threshold < 1:
            cv2.threshold(mask, self.threshold, 1, cv2.THRESH_BINARY, dst=mask)

        if self.postprocess:
            cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1, dst=mask)
            cv2.blur(mask, (10, 10), dst=mask)

        # Handle mask update speed
        bg_config = self.filters['background']
        if bg_config['mask_update_speed'] is not None:
            MRAR = getPercentageFloat(bg_config['mask_update_speed'])
            if MRAR < 1:
                if self.old_mask is None:
                    self.old_mask = mask
                mask = cv2.accumulateWeighted(mask, self.old_mask, MRAR)

        # Get background frame
        background_frame = None
        if self.images["background"] is None:
            # Default blur background
            blur_val = bg_config['blur'] if bg_config['blur'] is not None else 21
            blur_val = getNextOddNumber(blur_val)
            sigma = blur_val / 3
            background_frame = cv2.GaussianBlur(frame,
                                                (blur_val, blur_val),
                                                sigma,
                                                borderType=cv2.BORDER_DEFAULT)
        else:
            background_frame = next(self.images["background"])

        # Apply background effects
        background_frame = apply_effects_from_config(background_frame, bg_config)

        # Apply selfie effects
        frame = apply_effects_from_config(frame, self.filters['selfie'])

        # Replace background
        if self.use_sigmoid:
            mask = sigmoid(mask)

        # Handle opacity for selfie
        selfie_config = self.filters['selfie']
        if selfie_config['opacity'] is not None:
            opacity = min(100, max(0, int(selfie_config['opacity']))) / 100.0
            # Adjust the mask values by opacity (lower opacity = more background shows through)
            mask = mask * opacity

        cv2.blendLinear(frame, background_frame, mask, 1 - mask, dst=frame)

        # Add foreground mask if available
        if "foreground_mask" in self.images and "foreground" in self.images:
            # Apply mask effects to the foreground image
            foreground = self.images["foreground"].copy()
            foreground = apply_effects_from_config(foreground, self.filters['mask'])

            # Handle opacity for mask overlay
            mask_config = self.filters['mask']
            if mask_config['opacity'] is not None:
                opacity = min(100, max(0, int(mask_config['opacity']))) / 100.0
                # Adjust the mask values by opacity
                adjusted_mask = self.images["foreground_mask"] * opacity
                adjusted_inv_mask = 1 - adjusted_mask
                cv2.blendLinear(frame, foreground,
                        adjusted_inv_mask,
                        adjusted_mask, dst=frame)
            else:
                cv2.blendLinear(frame, foreground,
                        self.images["inverted_foreground_mask"],
                        self.images["foreground_mask"], dst=frame)

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
                if self.width != self.real_width or self.height != self.real_height:
                    frame = cv2.resize(frame, (self.width, self.height))
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
                    self.real_cam = None
                    if blank_image is not None:
                        blank_image.flags.writeable = True
                    blank_image = np.zeros((self.height, self.width), dtype=np.uint8)
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


def parser():
    parser = configargparse.ArgParser(
        description="Faking your webcam background under GNU/Linux. "
                    "Please refer to: https://github.com/fangfufu/Linux-Fake-Background-Webcam",
        epilog='''
Unified filter options for selfie, background, and mask:

Each component accepts a comma-separated list of effects:
- file=<filename>: Use specified image/video file
- hologram: Apply hologram effect
- blur=<N>: Apply blur with intensity 0-100
- solid=<B,G,R>: Fill with specific BGR color
- cmap=<name>: Apply color map using cmapy
- brightness=<N>: Adjust brightness 0-200 (100=normal)
- no: Disable component (background and mask)
- tile: Tile the image (background only)
- crop: Maintain aspect ratio by cropping (background only)
- mask-update-speed=<N>: Control mask update speed (background only)
- foreground=<filename>: Specify foreground image to overlay (mask only)
- mask-file=<filename>: Explicitly specify mask file when using foreground (mask only)
- opacity=<N>: Set opacity 0-100 (selfie: person transparency, mask: overlay transparency)

Examples:
%(prog)s --selfie=blur=30,hologram
%(prog)s --background=file=mybg.jpg,cmap=viridis
%(prog)s --background=no,blur=50
%(prog)s --mask=file=mask.png,blur=20
%(prog)s --mask=foreground=logo.png,mask-file=logo-mask.png,opacity=80
''',
        formatter_class=configargparse.RawTextHelpFormatter)

    parser.add_argument("-c", "--config", is_config_file=True,
                        help="Config file")
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
    parser.add_argument("--selfie", default="", type=str,
                        help="Selfie effects (comma-separated)")
    parser.add_argument("--background", default="file=background.jpg", type=str,
                        help="Background effects (comma-separated)")
    parser.add_argument("--mask", default="", type=str,
                        help="Mask effects (comma-separated)")
    parser.add_argument("--no-ondemand", action="store_true",
                        help="Continue processing when there is no application using the virtual webcam")
    parser.add_argument("--use-sigmoid", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--threshold", default="75", type=int,
                        help="The minimum percentage threshold for accepting a pixel as foreground")
    parser.add_argument("--no-postprocess", action="store_false",
                        help="Disable postprocessing (masking dilation and blurring)")
    parser.add_argument("--select-model", default="1", type=int,
                        help="Select the model for MediaPipe. For more information, please refer to "
                        "https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294")
    parser.add_argument("--dump", action="store_true",
                        help="Dump the filter configuration and exit")
    return parser


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


# Effect functions
def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 3, 4
    for y in range(holo.shape[0]):
        if y % (bandLength + bandGap) < bandLength:
            # Use deterministic value if TEST_DETERMINISTIC env var is set
            if os.environ.get('TEST_DETERMINISTIC'):
                holo[y, :, :] = holo[y, :, :] * 0.2  # Fixed value instead of random
            else:
                holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(
        holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(
        holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def blur_effect(frame, value=90):
    value = min(100, max(0, int(value)))
    if value == 0:
        return frame
    # Convert blur percentage to kernel size
    kernel_size = int((value / 100) * 99) + 1
    kernel_size = getNextOddNumber(kernel_size)
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def solid_effect(frame, color):
    frame[:] = color
    return frame

def cmap_effect(frame, map_name):
    cv2.applyColorMap(frame, cmap(map_name), dst=frame)
    return frame

def brightness_effect(frame, value=100):
    """Apply brightness effect by adjusting intensity."""
    brightness = min(200, max(0, int(value))) / 100.0  # Allow up to 200% brightness
    return np.clip(frame * brightness, 0, 255).astype(np.uint8)


def process_filter_args(filter_string):
    """Process comma-separated filter arguments into a list of [name, *args] lists."""
    if not filter_string:
        return []

    filters = []
    i = 0

    while i < len(filter_string):
        # Find the next equals sign
        eq_pos = filter_string.find('=', i)

        # Check for simple flags at current position
        for flag in ['no', 'hologram', 'tile', 'crop']:
            if filter_string[i:].startswith(flag):
                # Check if it's followed by comma or end of string
                flag_end = i + len(flag)
                if flag_end >= len(filter_string) or filter_string[flag_end] == ',':
                    filters.append([flag])
                    i = flag_end
                    if i < len(filter_string) and filter_string[i] == ',':
                        i += 1
                    continue

        if eq_pos == -1 or eq_pos < i:
            # No more equals signs
            remaining = filter_string[i:].strip()
            if remaining and remaining not in ['no', 'hologram', 'tile', 'crop']:
                print(f"Warning: Unknown filter '{remaining}'", file=sys.stderr)
            break

        # Get the filter name
        name = filter_string[i:eq_pos].strip()
        if i > 0 and filter_string[i-1] == ',':
            name = name[0:] if not name.startswith(',') else name[1:]

        # Find the value - need special handling for solid colors
        if name == 'solid':
            # For solid colors, we need to find 3 comma-separated numbers
            value_start = eq_pos + 1
            # Skip whitespace
            while value_start < len(filter_string) and filter_string[value_start].isspace():
                value_start += 1

            # Count commas to find BGR values
            comma_count = 0
            value_end = value_start
            while value_end < len(filter_string) and comma_count < 2:
                if filter_string[value_end] == ',':
                    comma_count += 1
                value_end += 1

            # Find the end of the third number
            while value_end < len(filter_string) and (filter_string[value_end].isdigit() or filter_string[value_end].isspace()):
                value_end += 1

            value = filter_string[value_start:value_end].strip()
            i = value_end
        else:
            # For other filters, find the next comma that's not inside a value
            value_start = eq_pos + 1
            next_comma = filter_string.find(',', value_start)

            # Check if there's another filter after this
            next_eq = filter_string.find('=', value_start)
            if next_eq != -1 and (next_comma == -1 or next_eq < next_comma):
                # There's another filter, find where this value ends
                j = next_eq - 1
                while j > value_start and filter_string[j].isspace():
                    j -= 1
                # Now backtrack to find the comma or start
                while j > value_start and filter_string[j] not in ',=':
                    j -= 1
                if filter_string[j] == ',':
                    value_end = j
                else:
                    value_end = j + 1
            elif next_comma != -1:
                value_end = next_comma
            else:
                value_end = len(filter_string)

            value = filter_string[value_start:value_end].strip()
            i = value_end

        # Skip comma if present
        if i < len(filter_string) and filter_string[i] == ',':
            i += 1

        # Process the filter
        if name == 'no':
            filters.append([name])
        elif name == 'hologram':
            filters.append([name])
        elif name == 'tile':
            filters.append([name])
        elif name == 'crop':
            filters.append([name])
        elif name == 'file' and value:
            filters.append([name, value])
        elif name == 'foreground' and value:
            filters.append([name, value])
        elif name == 'mask-file' and value:
            filters.append([name, value])
        elif name == 'opacity' and value:
            try:
                opacity_val = int(value)
                filters.append([name, opacity_val])
            except ValueError:
                print(f"Warning: Invalid opacity value '{value}'", file=sys.stderr)
        elif name == 'brightness' and value:
            try:
                brightness_val = int(value)
                filters.append([name, brightness_val])
            except ValueError:
                print(f"Warning: Invalid brightness value '{value}'", file=sys.stderr)
        elif name == 'cmap' and value:
            if value in colormaps:
                filters.append([name, value])
            else:
                print(f"Warning: Unknown colormap '{value}'", file=sys.stderr)
        elif name == 'blur' and value:
            try:
                blur_val = int(value)
                filters.append([name, blur_val])
            except ValueError:
                print(f"Warning: Invalid blur value '{value}'", file=sys.stderr)
        elif name == 'solid' and value:
            rgb = [int(e.strip()) if e.strip().isdigit() else 0
                   for e in value.split(',')[:3]]
            if len(rgb) == 3:
                filters.append([name, rgb])
            else:
                print(f"Warning: Invalid solid color '{value}'", file=sys.stderr)
        elif name == 'mask-update-speed' and value:
            try:
                speed = int(value)
                filters.append([name, speed])
            except ValueError:
                print(f"Warning: Invalid mask-update-speed '{value}'", file=sys.stderr)
        else:
            print(f"Warning: Unknown filter '{name}'", file=sys.stderr)

    return filters


def create_filter_config(filter_list, component_type):
    """Convert filter list to structured config dict with defaults."""
    # Define default configurations for each component type
    defaults = {
        'selfie': {
            'disabled': False,
            'file': None,
            'hologram': False,
            'blur': None,
            'solid': None,
            'cmap': None,
            'tile': False,
            'crop': False,
            'mask_update_speed': None,
            'foreground_file': None,
            'mask_file': None,
            'opacity': None,
            'brightness': None
        },
        'background': {
            'disabled': False,
            'file': 'background.jpg',  # Default background file
            'hologram': False,
            'blur': None,
            'solid': None,
            'cmap': None,
            'tile': False,
            'crop': False,
            'mask_update_speed': 50,  # Default mask update speed
            'foreground_file': None,
            'mask_file': None,
            'opacity': None,
            'brightness': None
        },
        'mask': {
            'disabled': False,
            'file': 'foreground-mask.png',
            'hologram': False,
            'blur': None,
            'solid': None,
            'cmap': None,
            'tile': False,
            'crop': False,
            'mask_update_speed': None,
            'foreground_file': 'foreground.jpg',
            'mask_file': 'foreground-mask.png',
            'opacity': None,
            'brightness': None
        }
    }

    # Start with defaults for the component type
    config = defaults[component_type].copy()

    # Apply filters from the list
    for filter_spec in filter_list:
        name = filter_spec[0]
        args = filter_spec[1:] if len(filter_spec) > 1 else []

        if name == 'no':
            config['disabled'] = True
            config['file'] = None  # Disable file when 'no' is specified
        elif name == 'hologram':
            config['hologram'] = True
        elif name == 'tile':
            config['tile'] = True
        elif name == 'crop':
            config['crop'] = True
        elif name == 'file' and args:
            config['file'] = args[0]
            config['disabled'] = False  # Enable when file is specified
        elif name == 'blur' and args:
            config['blur'] = args[0]
        elif name == 'solid' and args:
            config['solid'] = args[0]
        elif name == 'cmap' and args:
            config['cmap'] = args[0]
        elif name == 'mask-update-speed' and args:
            config['mask_update_speed'] = args[0]
        elif name == 'foreground' and args:
            config['foreground_file'] = args[0]
        elif name == 'mask-file' and args:
            config['mask_file'] = args[0]
            config['file'] = args[0]  # Also set file for backward compatibility
        elif name == 'opacity' and args:
            config['opacity'] = args[0]
        elif name == 'brightness' and args:
            config['brightness'] = args[0]

    return config


def apply_effects_from_config(frame, config):
    """Apply effects based on structured config dict."""
    if config['hologram']:
        frame = hologram_effect(frame)
    if config['blur'] is not None:
        frame = blur_effect(frame, config['blur'])
    if config['solid'] is not None:
        frame = solid_effect(frame, config['solid'])
    if config['cmap'] is not None:
        frame = cmap_effect(frame, config['cmap'])
    if config['brightness'] is not None:
        frame = brightness_effect(frame, config['brightness'])

    return frame


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


def get_codec_args_from_string(codec):
    return (char for char in codec)


def _log_camera_property_not_set(prop, value):
    print("Cannot set camera property {} to {}. "
          "Defaulting to auto-detected property set by opencv".format(prop,
                                                                      value))


def main():
    args = parser().parse_args()
    cam = FakeCam(args)
    signal.signal(signal.SIGINT, partial(sigint_handler, cam))
    signal.signal(signal.SIGQUIT, partial(sigquit_handler, cam))
    print("Running...")
    print("Please CTRL-C to pause and reload the background / foreground images")
    print("Please CTRL-\\ to exit")
    # frames forever
    cam.run()


if __name__ == "__main__":
    main()
