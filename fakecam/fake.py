import os
import cv2
import numpy as np
import requests
import pyfakewebcam
from signal import signal, SIGINT
from sys import exit

# setup access to the *real* webcam
cap = cv2.VideoCapture('/dev/video0')
height, width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)

# The scale factor for image sent to bodypix
sf = 0.5

# setup the fake camera
fake = pyfakewebcam.FakeWebcam('/dev/video2', width, height)

# declare global variables
background = None
foreground = None
f_mask = None
inv_f_mask = None

def load_images():
    global background
    global foreground
    global f_mask
    global inv_f_mask

    # load the virtual background
    background = cv2.imread("background.jpg")
    background = cv2.resize(background, (width, height))

    foreground = cv2.imread("foreground.jpg")
    foreground = cv2.resize(foreground, (width, height))

    f_mask = cv2.imread("foreground-mask.png")
    f_mask = cv2.normalize(f_mask, None, alpha=0, beta=1,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f_mask = cv2.resize(f_mask, (width, height))
    f_mask = cv2.cvtColor(f_mask, cv2.COLOR_BGR2GRAY)
    inv_f_mask = 1 - f_mask

def handler(signal_received, frame):
    load_images()
    print('Reloaded the background image')

def get_mask(frame, bodypix_url='http://127.0.0.1:9000'):
    frame = cv2.resize(frame, (0, 0), fx=sf, fy=sf)
    _, data = cv2.imencode(".png", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    mask = cv2.resize(mask, (0, 0), fx=1/sf, fy=1/sf,
                      interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((20,20), np.uint8) , iterations=1)
    mask = cv2.blur(mask.astype(float), (30,30))
    return mask

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

def get_frame(cap, background):
    _, frame = cap.read()
    # fetch the mask with retries (the app needs to warmup and we're lazy)
    # e v e n t u a l l y c o n s i s t e n t
    mask = None
    while mask is None:
        try:
            mask = get_mask(frame)
        except:
            print("mask request failed, retrying")

    # composite the foreground and background
    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c] * mask + background[:,:,c] * (1 - mask)

    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c] * inv_f_mask + foreground[:,:,c] * f_mask

    return frame

if __name__ == '__main__':
    load_images()
    signal(SIGINT, handler)
    print('Running...')
    print('Please press CTRL-\ to exit.')
    print('Please CTRL-C to reload the background and foreground images')
    # frames forever
    while True:
        frame = get_frame(cap, background)
        # fake webcam expects RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fake.schedule_frame(frame)
